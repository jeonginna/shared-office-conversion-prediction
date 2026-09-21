"""
공유오피스 결제 전환 예측 — 종합 대시보드 (ver2)
  Tab 1) 5개 모델 비교 (모델별 전처리 정확 구현, F2 등 6개 지표)
  Tab 2) XGBoost Optuna + SHAP 설명

[ver2 변경점]
* Optuna 버튼 3개: F2(메인) / 정확도 / Recall — 각각 다른 지표로 최적화
* max_depth 탐색 범위를 사용자가 슬라이더로 지정 (기본 2~8, 단순 모델 함정 방지)
* 탐색 범위를 노트북과 일치 (n_estimators 100~500, learning_rate 0.01~0.3, subsample 추가)
* 기본 trial 50

* no-show(visit_days==0) 제외 학습 · 모델별 전처리 다르게 적용 · 주지표 F2
실행: streamlit run office_dashboard_v2.py
"""
import warnings
warnings.filterwarnings("ignore")
import io
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform, os

# Matplotlib 한글 폰트 설정 (OS 자동 감지: Windows/Mac/Linux·Colab)
def _set_korean_font():
    system = platform.system()
    if system == "Windows":
        candidates = [r"C:\Windows\Fonts\malgun.ttf"]
    elif system == "Darwin":
        candidates = ["/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/AppleGothic.ttf"]
    else:
        candidates = ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]
        if not any(os.path.exists(p) for p in candidates):
            os.system("apt-get -qq -y install fonts-nanum > /dev/null 2>&1")
            try: fm._load_fontmanager(try_read_cache=False)
            except Exception: pass
    for p in candidates:
        if os.path.exists(p):
            try:
                fm.fontManager.addfont(p)
                plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            except Exception: pass
            break

_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    make_scorer, accuracy_score, roc_auc_score, f1_score,
    fbeta_score, precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay,
)

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

st.set_page_config(page_title="공유오피스 전환 예측", layout="wide")
st.title("🏢 공유오피스 고객 유료 전환 관리")
st.caption("관리자·분석가·실무자를 위한 결제 전환 분석 및 고객 관리 대시보드")

# ─────────────────────────────────────────────────────────────
# 컬럼 정의 (모델/전처리 가이드 기준)
# ─────────────────────────────────────────────────────────────
TARGET = "is_payment"
CAT_COLS = ["primary_site", "signup_dow"]
NUM_WITH_NA = ["primary_site_area", "last_visit_to_end_gap"]
NUM_TO_SCALE = ["stay_total_sec", "stay_mean_sec", "visit_days", "n_access_events",
                "n_sites_visited", "primary_site_area", "last_visit_to_end_gap"]
CLASS_NAMES = {0: "미결제", 1: "결제"}

# ── 직원용 예측 페이지용 메타데이터 ──
# 모델에 그대로 들어가는 수치형(원-핫 대상이 아닌) 컬럼
NUM_PASS = ["stay_total_sec", "stay_mean_sec", "visit_days", "n_access_events",
            "n_sites_visited", "primary_site_area", "last_visit_to_end_gap", "is_morning"]
# 지점 → 면적(평) 1:1 매핑 (site_dim.csv 기준) — 직원은 지점만 고르면 면적 자동 결정
# 원본 지점 마스터는 저장소에 포함하지 않는다 (앱과 같은 폴더의 site_dim.csv, .gitignore 대상)
SITE_AREA = (pd.read_csv("site_dim.csv")
             .set_index("site_id")["area_pyeong"].astype(int).to_dict())
SITE_IDS = list(SITE_AREA.keys())
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]   # signup_dow: 0=월 ~ 6=일 (dayofweek)
# 성과 추적: 직원이 캠페인 후 채워넣는 컬럼
TRACK_COLS = ["연락여부", "연락일자", "안내방식", "쿠폰제공여부", "실제결제여부", "결제일자"]
# 메일 발송 로그 파일명: 발송 성공/실패 기록을 남겨 캠페인 성과 추적에 연결
EMAIL_SEND_LOG = "email_send_log.csv"


def to_yes(v) -> bool:
    """'예/Y/1/O/완료' 등 다양한 입력을 True로 해석."""
    return str(v).strip().lower() in {"예", "y", "yes", "1", "o", "true", "t", "완료", "결제"}
# 피처 한글 라벨 (SHAP 설명·요약용)
FEATURE_KR = {
    "stay_total_sec": "총 체류시간", "stay_mean_sec": "방문당 평균 체류시간",
    "visit_days": "방문 일수", "n_access_events": "출입 횟수",
    "n_sites_visited": "이용 지점 수", "primary_site_area": "주 이용 지점 면적",
    "last_visit_to_end_gap": "마지막 방문~체험종료 간격", "is_morning": "오전 이용 여부",
    "primary_site": "주 이용 지점", "signup_dow": "가입 요일",
}



# ─────────────────────────────────────────────────────────────
# 메일 발송 헬퍼 함수
# ─────────────────────────────────────────────────────────────
def get_email_credentials():
    """Streamlit secrets에서 SMTP 계정 정보를 읽는다.
    .streamlit/secrets.toml 예시:
    [email]
    sender = "your_email@gmail.com"
    app_password = "gmail_app_password"
    smtp_host = "smtp.gmail.com"
    smtp_port = 465
    """
    try:
        cfg = st.secrets["email"]
        return {
            "sender": cfg["sender"],
            "app_password": cfg["app_password"],
            "smtp_host": cfg.get("smtp_host", "smtp.gmail.com"),
            "smtp_port": int(cfg.get("smtp_port", 465)),
        }
    except Exception:
        return None


def build_campaign_email_body(row, benefit_text):
    """고객별 예측 정보로 메일 본문을 만든다.
    실제 운영에서는 브랜드 톤에 맞춰 이 템플릿만 다듬으면 된다.
    """
    if "본문" in row and pd.notna(row["본문"]) and str(row["본문"]).strip():
        return str(row["본문"]).replace("\n", "<br>")

    site = row.get("primary_site", "이용 지점")
    visit_days = int(row.get("visit_days", 0)) if pd.notna(row.get("visit_days", 0)) else 0
    stay_hours = row.get("stay_total_sec", 0) / 3600 if pd.notna(row.get("stay_total_sec", 0)) else 0
    proba_pct = row.get("결제확률(%)", row.get("pred_proba", 0) * 100)

    return f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.7; color: #222;">
      <p>안녕하세요. 공유오피스 3일 체험을 이용해주셔서 감사합니다.</p>

      <p>
        고객님은 체험 기간 동안 <b>{site}번 지점</b>을 중심으로
        <b>{visit_days}일</b> 방문해주셨고, 총 <b>{stay_hours:.1f}시간</b> 이용해주셨습니다.
      </p>

      <p>
        체험 이용 패턴을 바탕으로 정기 이용 전환 가능성이 높은 고객님께
        아래 혜택을 먼저 안내드립니다.
      </p>

      <p style="padding: 12px; background: #f3f6ff; border-radius: 8px;">
        <b>추천 혜택:</b> {benefit_text}
      </p>

      <p>
        업무 공간이 다시 필요하시다면 이번 혜택을 통해 더 합리적으로 이용해보세요.
      </p>

      <p style="font-size: 12px; color: #666;">
        내부 예측 점수: {proba_pct:.1f}%<br>
        본 메일은 3일 체험 이용 고객 대상 안내 메일입니다.
      </p>
    </div>
    """


def get_campaign_subject(row, default_subject):
    """CSV에 고객별 제목이 있으면 그 제목을 쓰고, 없으면 공통 제목을 쓴다."""
    if "제목" in row and pd.notna(row["제목"]) and str(row["제목"]).strip():
        return str(row["제목"])
    return default_subject


def send_campaign_email(to_email, subject, body_html):
    """SMTP로 HTML 메일을 발송한다. 계정 정보는 st.secrets에 둔다."""
    creds = get_email_credentials()
    if creds is None:
        raise RuntimeError("메일 계정 설정이 없습니다. .streamlit/secrets.toml에 [email] 정보를 추가하세요.")

    msg = MIMEText(body_html, "html", "utf-8")
    msg["Subject"] = subject
    msg["From"] = creds["sender"]
    msg["To"] = to_email

    with smtplib.SMTP_SSL(creds["smtp_host"], creds["smtp_port"]) as smtp:
        smtp.login(creds["sender"], creds["app_password"])
        smtp.send_message(msg)


def append_email_log(log_rows):
    """발송 결과를 CSV로 누적 저장한다."""
    if not log_rows:
        return
    log_df = pd.DataFrame(log_rows)
    if os.path.exists(EMAIL_SEND_LOG):
        prev = pd.read_csv(EMAIL_SEND_LOG)
        log_df = pd.concat([prev, log_df], ignore_index=True)
    log_df.to_csv(EMAIL_SEND_LOG, index=False, encoding="utf-8-sig")

def feature_kr(col: str) -> str:
    """원-핫 더미명(primary_site_3.0, signup_dow_1 등)을 한글 라벨로 변환."""
    if col.startswith("primary_site_"):
        sid = col.rsplit("_", 1)[1].replace(".0", "")
        return f"주 이용 지점({sid}번)"
    if col.startswith("signup_dow_"):
        d = int(col.rsplit("_", 1)[1])
        return f"가입 요일({DOW_KR[d]})"
    return FEATURE_KR.get(col, col)


def build_input_row(d: dict, train_cols: list) -> pd.DataFrame:
    """직원 입력 dict → 학습 컬럼 구조의 1행 DataFrame.
    dict→DataFrame 변환 시 정수 범주가 float로 바뀌어 더미명이 어긋나는 문제를
    피하려고, 0벡터에서 해당 더미만 직접 점등한다(전처리 일관성 보장)."""
    v = pd.Series(0.0, index=train_cols)
    for c in NUM_PASS:
        if c in v.index:
            v[c] = d[c]
    ps = "none" if pd.isna(d["primary_site"]) else str(float(d["primary_site"]))
    if (col := f"primary_site_{ps}") in v.index:
        v[col] = 1.0
    if (col := f"signup_dow_{int(d['signup_dow'])}") in v.index:
        v[col] = 1.0
    return pd.DataFrame([v])[train_cols]


# ─────────────────────────────────────────────────────────────
# 데이터 로드 (no-show 제외)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("df_master_output.csv")
    df_full = df.copy()
    df_model = df[df["visit_days"] > 0].reset_index(drop=True)
    return df_full, df_model


@st.cache_data
def get_split(test_size):
    _, df_model = load_data()
    X = df_model.drop(columns=TARGET)
    y = df_model[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


df_full, df_model = load_data()


# ─────────────────────────────────────────────────────────────
# 평가 지표 (6개)
# ─────────────────────────────────────────────────────────────
def all_metrics(y_true, pred, proba):
    return {
        "F2": round(fbeta_score(y_true, pred, beta=2), 3),
        "AUC": round(roc_auc_score(y_true, proba), 3),
        "Recall": round(recall_score(y_true, pred), 3),
        "Precision": round(precision_score(y_true, pred), 3),
        "F1": round(f1_score(y_true, pred), 3),
        "정확도": round(accuracy_score(y_true, pred), 3),
    }


# ─────────────────────────────────────────────────────────────
# 모델별 전처리 (가이드 기준, 모델마다 다름) · 모든 fit은 train에서만
# ─────────────────────────────────────────────────────────────
def prep_logreg(X_train, X_test):
    tr, te = X_train.copy(), X_test.copy()
    for d in (tr, te):
        d["primary_site"] = d["primary_site"].fillna("none").astype(str)
    for c in NUM_WITH_NA:
        med = tr[c].median()
        for d in (tr, te):
            d[c + "_missing"] = d[c].isna().astype(int)
            d[c] = d[c].fillna(med)
    tr = pd.get_dummies(tr, columns=CAT_COLS)
    te = pd.get_dummies(te, columns=CAT_COLS)
    tr, te = tr.align(te, join="left", axis=1, fill_value=0)
    scaler = StandardScaler()
    tr[NUM_TO_SCALE] = scaler.fit_transform(tr[NUM_TO_SCALE])
    te[NUM_TO_SCALE] = scaler.transform(te[NUM_TO_SCALE])
    return tr, te


def prep_tree(X_train, X_test):
    tr, te = X_train.copy(), X_test.copy()
    for d in (tr, te):
        d["primary_site"] = d["primary_site"].fillna("none").astype(str)
        for c in NUM_WITH_NA:
            d[c] = d[c].fillna(-1)
    tr = pd.get_dummies(tr, columns=CAT_COLS)
    te = pd.get_dummies(te, columns=CAT_COLS)
    tr, te = tr.align(te, join="left", axis=1, fill_value=0)
    return tr, te


def prep_boost(X_train, X_test):
    tr, te = X_train.copy(), X_test.copy()
    for d in (tr, te):
        d["primary_site"] = d["primary_site"].fillna("none").astype(str)
    tr = pd.get_dummies(tr, columns=CAT_COLS)
    te = pd.get_dummies(te, columns=CAT_COLS)
    tr, te = tr.align(te, join="left", axis=1, fill_value=0)
    return tr, te


def prep_cat(X_train, X_test):
    tr, te = X_train.copy(), X_test.copy()
    for d in (tr, te):
        d["primary_site"] = d["primary_site"].fillna("none").astype(str)
        d["signup_dow"] = d["signup_dow"].astype(str)
    return tr, te


# ─────────────────────────────────────────────────────────────
# 모델 비교 (불균형 보정 토글)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def compare_models(test_size, balanced):
    X_train, X_test, y_train, y_test = get_split(test_size)
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    rows, cms = [], {}

    tr, te = prep_logreg(X_train, X_test)
    cw = "balanced" if balanced else None
    m = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42).fit(tr, y_train)
    pred, proba = m.predict(te), m.predict_proba(te)[:, 1]
    rows.append({"모델": "Logistic Regression", **all_metrics(y_test, pred, proba)})
    cms["Logistic Regression"] = confusion_matrix(y_test, pred)

    tr, te = prep_tree(X_train, X_test)
    cw = "balanced" if balanced else None
    m = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                               class_weight=cw, random_state=42).fit(tr, y_train)
    pred, proba = m.predict(te), m.predict_proba(te)[:, 1]
    rows.append({"모델": "Decision Tree", **all_metrics(y_test, pred, proba)})
    cms["Decision Tree"] = confusion_matrix(y_test, pred)

    tr, te = prep_boost(X_train, X_test)
    kw = {"scale_pos_weight": spw} if balanced else {}
    m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                          eval_metric="logloss", random_state=42, **kw).fit(tr, y_train)
    pred, proba = m.predict(te), m.predict_proba(te)[:, 1]
    rows.append({"모델": "XGBoost", **all_metrics(y_test, pred, proba)})
    cms["XGBoost"] = confusion_matrix(y_test, pred)

    if HAS_LGBM:
        tr, te = prep_boost(X_train, X_test)
        cw = "balanced" if balanced else None
        m = LGBMClassifier(n_estimators=300, learning_rate=0.1,
                           class_weight=cw, random_state=42, verbose=-1).fit(tr, y_train)
        pred, proba = m.predict(te), m.predict_proba(te)[:, 1]
        rows.append({"모델": "LightGBM", **all_metrics(y_test, pred, proba)})
        cms["LightGBM"] = confusion_matrix(y_test, pred)

    if HAS_CAT:
        tr, te = prep_cat(X_train, X_test)
        acw = "Balanced" if balanced else None
        m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1,
                               auto_class_weights=acw,
                               cat_features=["primary_site", "signup_dow"],
                               verbose=0, random_state=42).fit(tr, y_train)
        pred, proba = m.predict(te), m.predict_proba(te)[:, 1]
        rows.append({"모델": "CatBoost", **all_metrics(y_test, pred, proba)})
        cms["CatBoost"] = confusion_matrix(y_test, pred)

    result = pd.DataFrame(rows).sort_values("F2", ascending=False).reset_index(drop=True)
    return result, cms


# ─────────────────────────────────────────────────────────────
# XGBoost 학습 (Tab 2용, 수동 파라미터)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def train_xgb(n_estimators, max_depth, learning_rate, test_size, balanced):
    X_train, X_test, y_train, y_test = get_split(test_size)
    tr, te = prep_boost(X_train, X_test)
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    kw = {"scale_pos_weight": spw} if balanced else {}
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
        eval_metric="logloss", random_state=42, **kw,
    )
    model.fit(tr, y_train)
    return model, tr, te, y_train, y_test


# ─────────────────────────────────────────────────────────────
# 직원용 예측 (현재 사이드바 설정의 XGB 재사용) · 전체 고객 점수도 함께 캐싱
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_predictor(n_estimators, max_depth, learning_rate, test_size, balanced):
    """직원 페이지 전용: 학습된 모델 + 학습 컬럼 + SHAP explainer +
    전체 학습대상 고객 결제확률(타겟 리스트용)을 한 번에 만들어 캐싱."""
    model, tr, te, y_train, y_test = train_xgb(
        n_estimators, max_depth, learning_rate, test_size, balanced)
    train_cols = tr.columns.tolist()
    explainer = shap.TreeExplainer(model)

    # 전체 학습대상(no-show 제외) 결제확률 — 타겟 리스트(배치)용
    _, df_model_local = load_data()
    X_all = prep_boost(df_model_local.drop(columns=TARGET), df_model_local.drop(columns=TARGET))[0]
    X_all = X_all.reindex(columns=train_cols, fill_value=0)
    proba_all = model.predict_proba(X_all)[:, 1]
    scored = df_model_local.copy()
    scored["pred_proba"] = proba_all
    scored.insert(0, "고객번호", [f"#{i + 1:04d}" for i in range(len(scored))])
    return model, train_cols, explainer, scored


def predict_frame(df_up, model, train_cols):
    """여러 고객(DataFrame)을 한 번에 예측 — 학습과 동일한 prep_boost 경로(벡터화)."""
    X = df_up.copy()
    X["primary_site"] = X["primary_site"].fillna("none").astype(str)
    X["signup_dow"] = X["signup_dow"].astype(int)          # 정수 범주 고정(더미명 일치)
    Xd = pd.get_dummies(X[[*NUM_PASS, *CAT_COLS]], columns=CAT_COLS)
    Xd = Xd.reindex(columns=train_cols, fill_value=0)
    return model.predict_proba(Xd)[:, 1]


@st.cache_data(show_spinner="고객 명단 예측 중…")
def predict_uploaded(file_bytes, settings):
    """업로드 CSV → 결제확률 부여. 파일 내용+모델설정을 키로 캐싱(재계산 방지)."""
    model, train_cols, _, _ = get_predictor(*settings)
    df_up = pd.read_csv(io.BytesIO(file_bytes))
    need = [*NUM_PASS, *CAT_COLS]
    missing = [c for c in need if c not in df_up.columns]
    if missing:
        return None, missing, []
    id_cols = [c for c in df_up.columns if c not in need and c != TARGET]  # 식별자 등 보존
    out = df_up.copy()
    out["pred_proba"] = predict_frame(df_up, model, train_cols)
    return out, [], id_cols


def prepare_prescored_mail_file(file_bytes):
    """이미 결제확률·메일 내용이 들어있는 CSV를 발송 명단 형태로 변환한다.
    예: 이메일, 고객명, 결제확률, 제목, 본문 컬럼이 있는 파일.
    """
    df_mail = pd.read_csv(io.BytesIO(file_bytes))
    rename_map = {}
    if "이메일" in df_mail.columns and "email" not in df_mail.columns:
        rename_map["이메일"] = "email"
    if "고객명" in df_mail.columns and "고객번호" not in df_mail.columns:
        rename_map["고객명"] = "고객번호"
    df_mail = df_mail.rename(columns=rename_map)

    if "email" not in df_mail.columns:
        return None

    if "pred_proba" not in df_mail.columns:
        if "결제확률" in df_mail.columns:
            prob = pd.to_numeric(df_mail["결제확률"], errors="coerce")
            df_mail["pred_proba"] = np.where(prob > 1, prob / 100, prob)
        elif "결제확률(%)" in df_mail.columns:
            prob = pd.to_numeric(df_mail["결제확률(%)"], errors="coerce")
            df_mail["pred_proba"] = np.where(prob > 1, prob / 100, prob)
        else:
            return None

    df_mail["pred_proba"] = pd.to_numeric(df_mail["pred_proba"], errors="coerce").fillna(0)
    df_mail["결제확률(%)"] = (df_mail["pred_proba"] * 100).round(1)
    id_cols = [c for c in ["고객번호", "email", "세그먼트", "제목", "본문"] if c in df_mail.columns]
    return df_mail, id_cols


# ─────────────────────────────────────────────────────────────
# Optuna — 지표별 버튼(F2 / 정확도 / Recall)
# ─────────────────────────────────────────────────────────────
F2_SCORER = make_scorer(fbeta_score, beta=2)
METRIC_SCORERS = {
    "F2": F2_SCORER,          # 메인 (Recall 가중)
    "정확도": "accuracy",
    "Recall": "recall",
}


def make_run_optuna(metric_name):
    """지표 이름을 받아, 그 지표로 최적화하는 버튼 콜백을 생성한다."""
    def _callback():
        st.cache_resource.clear()
        test_size = st.session_state.get("test_size", 0.2)
        n_trials = st.session_state.get("n_trials", 50)
        balanced = st.session_state.get("balanced", True)
        depth_min, depth_max = st.session_state.get("depth_range", (2, 8))  # 사용자 지정 범위

        X_train, X_test, y_train, y_test = get_split(test_size)
        tr, _ = prep_boost(X_train, X_test)
        spw = (y_train == 0).sum() / (y_train == 1).sum()
        scoring = METRIC_SCORERS[metric_name]

        def objective(trial):
            params = {
                # n_estimators·learning_rate·subsample는 노트북과 동일 범위 (단순 모델 함정 방지)
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", depth_min, depth_max),  # 사용자 범위
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            kw = {"scale_pos_weight": spw} if balanced else {}
            model = xgb.XGBClassifier(**params, **kw, eval_metric="logloss", random_state=42)
            return cross_val_score(model, tr, y_train, cv=3, scoring=scoring).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        bp = study.best_params
        st.session_state["n_estimators"] = bp["n_estimators"]
        st.session_state["max_depth"] = bp["max_depth"]
        st.session_state["learning_rate"] = round(bp["learning_rate"], 2)
        st.session_state["study"] = study
        st.session_state["best_cv_score"] = study.best_value
        st.session_state["opt_metric"] = metric_name      # 어느 지표로 최적화했는지
    return _callback


# ─────────────────────────────────────────────────────────────
# 위젯 기본값
# ─────────────────────────────────────────────────────────────
DEFAULTS = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.10,
            "test_size": 0.20, "n_trials": 50, "balanced": True, "depth_range": (2, 8)}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚖️ 공통 설정")
st.sidebar.toggle("불균형 보정 (class_weight) 적용", key="balanced",
                  help="켜면 5개 모델 전부에 불균형 보정을 적용합니다. (F2/Recall 개선)")
st.sidebar.slider("test_size", 0.1, 0.5, step=0.05, key="test_size")

st.sidebar.divider()
st.sidebar.header("🤖 Optuna 자동 탐색 (XGB)")
st.sidebar.slider("trial 횟수", 10, 200, step=10, key="n_trials")
# max_depth 탐색 범위 (사용자 지정) — 단순 모델 함정 방지의 핵심 노브
st.sidebar.slider("max_depth 탐색 범위", 1, 12, key="depth_range",
                  help="Optuna가 탐색할 max_depth의 최소~최대. 너무 낮게(1) 열면 과도하게 단순한 모델이 나올 수 있습니다. 권장 2~8.")

st.sidebar.caption("**지표별 최적화** — 메인은 F2")
st.sidebar.button("🎯 F2 기준 최적화 (메인)", on_click=make_run_optuna("F2"),
                  use_container_width=True, type="primary")
b1, b2 = st.sidebar.columns(2)
with b1:
    st.button("정확도 기준", on_click=make_run_optuna("정확도"), use_container_width=True)
with b2:
    st.button("Recall 기준", on_click=make_run_optuna("Recall"), use_container_width=True)

if "best_cv_score" in st.session_state:
    metric = st.session_state.get("opt_metric", "F2")
    st.sidebar.success(f"[{metric} 최적화] best CV {metric}: {st.session_state['best_cv_score']:.3f}")

st.sidebar.divider()
st.sidebar.header("⚙️ XGB 하이퍼파라미터 (수동)")
st.sidebar.caption("Optuna가 채워준 값에서 직접 흔들어보세요.")
st.sidebar.slider("n_estimators", 50, 500, step=50, key="n_estimators")
st.sidebar.slider("max_depth", 1, 12, key="max_depth")
st.sidebar.slider("learning_rate", 0.01, 0.5, step=0.01, key="learning_rate")


# ─────────────────────────────────────────────────────────────
# 렌더 헬퍼
# ─────────────────────────────────────────────────────────────
def render(fig_fn):
    plt.close("all")
    fig_fn()
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.close("all")


def plot_cm(cm, title):
    tn, fp, fn, tp = cm.ravel()
    disp = ConfusionMatrixDisplay(cm, display_labels=["미결제(0)", "결제(1)"])
    plt.close("all")
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    st.pyplot(plt.gcf(), bbox_inches="tight")
    plt.close("all")
    rec = tp / (tp + fn) if (tp + fn) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    st.caption(
        f"실제 결제자 {tp+fn}명 중 {tp}명을 잡아 Recall {rec:.1%}. "
        f"결제로 예측한 {tp+fp}명 중 {tp}명이 실제 결제자라 Precision {prec:.1%}. "
        f"→ 결제자를 {rec:.1%} 잡아내지만 미결제자 {fp}명(FP)에게도 메일이 발송됨."
    )


# ─────────────────────────────────────────────────────────────
# [메일 CSV 생성] 부록참고 노트북의 발송파일 생성 로직 (상위 N%)
#   업로드 CSV → 모델 채점 → 미결제·체험종료 → 상위 N% → 세그먼트/제목/본문
#   → df_master_mail_ver2.csv 양식으로 반환 (n8n이 이 CSV를 먹음)
#   ※ 날짜는 to_datetime을 쓰지 않고 문자열로 비교(일부 환경 크래시 회피).
# ─────────────────────────────────────────────────────────────
MAIL_PHRASE = {
    "last_visit_to_end_gap": "체험 마지막 날까지 알차게 이용해 주셨네요.",
    "visit_days": "짧은 체험 기간에도 여러 번 찾아주셨어요.",
    "stay_total_sec": "체험 기간 동안 공간을 여유롭게 이용해 주셨어요.",
    "stay_mean_sec": "방문하실 때마다 여유 있게 머무르셨어요.",
    "n_access_events": "체험 기간 동안 공간을 활발히 이용해 주셨어요.",
    "n_sites_visited": "여러 지점을 둘러보며 체험해 주셨네요.",
}


def _mail_make_email(r, deadline, sign):
    """세그먼트별 제목/본문 생성 (부록참고 노트북과 동일)."""
    s = r["primary_site"]
    jijeom = "이용하신 지점" if pd.isna(s) else f"{int(float(s))}번 지점"
    area = "" if pd.isna(r.get("primary_site_area")) else f"({int(float(r['primary_site_area']))}평)"
    name = r["고객명"]
    reason = (r["근거문구"] + "\n") if r["근거문구"] else ""
    if r["세그먼트"] == "높음":
        subj = f"(광고) {name}님, {jijeom} 첫 달 30% 혜택 곧 마감돼요"
        body = (f"{name}님, 안녕하세요 :) {jijeom}{area} 체험은 잘 마치셨나요?\n{reason}"
                f"지금 멤버십을 등록하시면 첫 달 30% 할인을 드려요. "
                f"이 혜택은 {deadline}까지만 열려 있어요.\n[지금 등록하고 할인받기]{sign}")
    else:
        subj = f"(광고) {name}님, {jijeom} 한 번 더 둘러보실래요?"
        body = (f"{name}님, 안녕하세요 :) {jijeom}{area} 체험은 어떠셨나요?\n{reason}"
                f"아직 결정이 어려우시다면, 부담 없이 한 번 더 방문해보세요. "
                f"재방문 시 사용할 수 있는 1일 무료 이용 쿠폰을 추가로 드려요.\n[쿠폰 받고 다시 방문하기]{sign}")
    return pd.Series({"제목": subj, "본문": body})


def _plus_addr(base, uuid):
    """입력 이메일(base)에 uuid를 플러스 어드레싱. base에 @없으면 gmail.com 가정."""
    base = str(base).strip()
    if "@" in base:
        local, domain = base.split("@", 1)
    else:
        local, domain = base, "gmail.com"
    return f"{local}+{uuid}@{domain}"


@st.cache_data(show_spinner="사이드바 설정 모델로 채점 · 메일 파일 생성 중…")
def build_mail_table(file_bytes, top_pct, n_estimators, max_depth, learning_rate,
                     test_size, balanced, base_email=""):
    from datetime import datetime, timedelta
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.lstrip("\ufeff") for c in df.columns]     # BOM 제거

    need = NUM_PASS + CAT_COLS + [TARGET, "user_uuid", "trial_end"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise KeyError("업로드 CSV에 필요한 컬럼이 없습니다: " + ", ".join(missing))

    # 1) 사이드바(옵튜나·수동 슬라이더)로 학습된 모델·SHAP explainer를 그대로 가져온다
    #    → 4번 탭이 따로 고정 학습하지 않고, 사이드바에서 최적화한 그 모델로 채점.
    model, train_cols, explainer, _scored = get_predictor(
        n_estimators, max_depth, learning_rate, test_size, balanced)

    # 2) 업로드 CSV를 그 모델의 학습 컬럼에 맞춰 원-핫 정렬 후 채점 (no-show 제외)
    work = df[df["visit_days"] > 0].reset_index(drop=True).copy()
    Xraw = work[NUM_PASS + CAT_COLS].copy()
    Xraw["primary_site"] = Xraw["primary_site"].fillna("none").astype(str)
    X = pd.get_dummies(Xraw, columns=CAT_COLS).reindex(columns=train_cols, fill_value=0)
    work["결제확률"] = model.predict_proba(X)[:, 1]

    # 2) 미결제 & 체험종료(오늘 이전) → 상위 N% (날짜는 문자열 비교)
    work["_te"] = work["trial_end"].astype(str).str[:10]
    today_s = datetime.now().strftime("%Y-%m-%d")
    eligible = work[(work[TARGET] == 0) & (work["_te"] < today_s)].copy()
    n_target = max(1, int(len(eligible) * top_pct / 100))
    mail = eligible.nlargest(n_target, "결제확률").copy()

    # 3) 세그먼트 (중앙값 기준 높음/중간)
    med = mail["결제확률"].median()
    mail["세그먼트"] = np.where(mail["결제확률"] >= med, "높음", "중간")

    # 4) SHAP 근거문구 (get_predictor가 준 explainer 사용, 실패해도 빈 문자열로 진행)
    try:
        Xr = X.loc[mail.index]
        sv = explainer(Xr).values
        feats = [f for f in MAIL_PHRASE if f in Xr.columns]
        cols = [Xr.columns.get_loc(f) for f in feats]
        contrib = sv[:, cols]
        best = contrib.argmax(axis=1)
        strongest = contrib.max(axis=1)
        mail["근거문구"] = np.where(strongest > 0,
                                 np.array([MAIL_PHRASE[feats[b]] for b in best]), "")
    except Exception:
        mail["근거문구"] = ""

    # 5) 고객명 / 이메일
    #    - 입력창(base_email) 채우면 → 그 이메일에 uuid 플러스 어드레싱 (email 컬럼 유무 무관)
    #    - 입력창 비우고 email 컬럼 있으면 → CSV email 그대로
    #    - 입력창 비우고 email 컬럼 없으면 → 데모 이메일
    mail = mail.reset_index(drop=True)
    seq = (mail.index + 1).astype(str).str.zfill(4)
    mail["고객명"] = "고객" + seq
    base = str(base_email or "").strip()
    has_email_col = ("email" in mail.columns) and mail["email"].notna().any()
    if base:
        mail["이메일"] = mail["user_uuid"].astype(str).map(lambda u: _plus_addr(base, u))
    elif has_email_col:
        mail["이메일"] = mail["email"].astype(str)
    else:
        mail["이메일"] = "test+" + seq + "@example.com"

    # 6) 제목/본문
    _dl = datetime.now() + timedelta(days=7)
    deadline = f"{_dl.month}월 {_dl.day}일"
    sign = ("\n──────────\n본 메일은 광고입니다 · 공유 오피스 OOO\n"
            "무료수신거부 ▸ [링크]  |  문의 ▸ [연락처]")
    mail[["제목", "본문"]] = mail.apply(lambda r: _mail_make_email(r, deadline, sign), axis=1)

    # 7) 운영 컬럼 + 중복방지 키(trial_key)
    if "trial_key" not in mail.columns:
        mail["trial_key"] = (mail["user_uuid"].astype(str) + "_"
                             + mail["trial_end"].astype(str).str[:10])
    if "last_visit_date" not in mail.columns:
        mail["last_visit_date"] = ""
    mail["trial_date"] = mail.get("trial_date", "").astype(str).str[:10] if "trial_date" in mail.columns else ""
    mail["trial_end"] = mail["trial_end"].astype(str).str[:10]
    mail["scored_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mail["campaign_id"] = f"trial_conversion_top{top_pct}_v1"
    mail["send_yn"] = "Y"
    mail["발송상태"] = "대기"

    out_cols = ["이메일", "고객명", "세그먼트", "결제확률", "제목", "본문", TARGET,
                "trial_key", "user_uuid", "trial_date", "trial_end", "last_visit_date",
                "primary_site", "scored_at", "campaign_id", "send_yn", "발송상태"]
    out_cols = [c for c in out_cols if c in mail.columns]
    result = mail[out_cols].copy()
    result["결제확률"] = (result["결제확률"] * 100).round(1)

    info = {"eligible": int(len(eligible)), "n_mail": int(len(result)),
            "seg": result["세그먼트"].value_counts().to_dict(),
            "key_unique": bool(result["trial_key"].is_unique) if "trial_key" in result else True,
            "all_unpaid": bool((result[TARGET] == 0).all())}
    return result, info


# ─────────────────────────────────────────────────────────────
# 탭
# ─────────────────────────────────────────────────────────────
tab_cmp, tab_opt, tab_use, tab_mail = st.tabs(
    ["📊 모델 비교", "🔬 Optuna · SHAP (XGBoost)", "💼 결제 전환 예측 (직원용)",
     "📧 메일 CSV 생성 (n8n용)"])

# ============================ Tab 1 ============================
with tab_cmp:
    st.subheader("5개 모델 성능 비교")
    bal_txt = "적용" if st.session_state["balanced"] else "미적용"
    st.caption(f"no-show 제외 학습 · 모델별 전처리 정확 구현 · 불균형 보정 **{bal_txt}** · 주지표 **F2**")

    result, cms = compare_models(st.session_state["test_size"], st.session_state["balanced"])

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.dataframe(result, use_container_width=True, hide_index=True)
    with c2:
        st.bar_chart(result.set_index("모델")["F2"])

    best = result.iloc[0]
    st.info(
        f"**F2 1위: {best['모델']}** (F2 {best['F2']:.3f}). "
        "메일 프로모션은 '놓치지 않기(Recall)'가 중요해 F2를 주지표로 봅니다. "
        "단, F2가 높아도 Precision이 낮으면 헛메일(FP)이 많을 수 있어 혼동행렬을 함께 확인하세요."
    )

    with st.expander("🧩 혼동행렬 (모델별)"):
        sel = st.selectbox("모델 선택", list(cms.keys()))
        plot_cm(cms[sel], f"혼동행렬 — {sel}")

    with st.expander("📋 데이터 미리보기 · 전환율"):
        st.write(f"**전체 {len(df_full):,}명 / 학습 대상(no-show 제외) {len(df_model):,}명** "
                 f"· 전환율 {df_model[TARGET].mean():.1%}")
        st.dataframe(df_model.head(10), use_container_width=True)
        e1, e2 = st.columns(2)
        with e1:
            st.write("**주 지점(primary_site)별 전환율**")
            st.dataframe(df_model.groupby("primary_site")[TARGET].mean().round(3))
        with e2:
            st.write("**방문일수(visit_days)별 전환율**")
            st.dataframe(df_model.groupby("visit_days")[TARGET].mean().round(3))

    with st.expander("🛠️ 전처리 · 인코딩 · 스케일링 현황 (모델/전처리 가이드 기준)"):
        st.markdown("""
**공통 원칙**
- **no-show(visit_days=0) 제외** 후 학습 — 변수가 없어 예측 불가(사전 실험 결론).
- 전처리 기준값(중앙값·스케일러)은 **train에서만 fit** → 데이터 누수 방지.
- 불균형 보정: 사이드바 토글 (로지스틱 `class_weight`, XGB `scale_pos_weight`, CatBoost `auto_class_weights`).

**모델별 전처리**

| 모델 | primary_site / signup_dow | 수치형 결측 | 스케일링 |
|---|---|---|---|
| Logistic Regression | `'none'` + 원-핫 | **중앙값 + 미방문 플래그** | **StandardScaler (수치형만)** |
| Decision Tree | `'none'` + 원-핫 | **-1 (특수값)** | 불필요 |
| XGBoost | `'none'` + 원-핫 | **NaN 그대로** (자동 처리) | 불필요 |
| LightGBM | `'none'` + 원-핫 | **NaN 그대로** (자동 처리) | 불필요 |
| CatBoost | **cat_features 직접** (원-핫 X) | **NaN 그대로** (자동 처리) | 불필요 |

**왜 다른가**
- 로지스틱(선형)은 스케일·결측에 민감 → 중앙값+플래그+스케일링 필요. `-1`은 "작은 값"으로 왜곡되어 금지.
- 트리 계열은 순서로만 분할 → 스케일링 불필요. DT는 NaN을 못 받아 `-1`, 부스팅은 NaN 자동 처리.
- CatBoost는 범주형을 직접 받아 원-핫 불필요.
""")

# ============================ Tab 2 ============================
with tab_opt:
    st.subheader("XGBoost — Optuna 최적화 & SHAP 설명")
    st.caption("사이드바에서 지표별 버튼(F2/정확도/Recall)으로 Optuna를 실행하면 슬라이더가 best params로 스냅됩니다.")

    model, tr, te, y_train, y_test = train_xgb(
        st.session_state["n_estimators"], st.session_state["max_depth"],
        st.session_state["learning_rate"], st.session_state["test_size"],
        st.session_state["balanced"],
    )
    pred = model.predict(te)
    proba = model.predict_proba(te)[:, 1]

    mc = all_metrics(y_test, pred, proba)
    cols = st.columns(6)
    for col, (k, v) in zip(cols, mc.items()):
        col.metric(k, f"{v:.3f}")

    if "study" in st.session_state:
        metric = st.session_state.get("opt_metric", "F2")
        st.markdown(f"#### 📈 Optuna 탐색 결과 ({metric} 기준)")
        study = st.session_state["study"]
        st.write(f"**best params** ({metric} 최적화) → {study.best_params}")
        try:
            import optuna.visualization as vis
            o1, o2 = st.columns(2)
            with o1:
                st.plotly_chart(vis.plot_optimization_history(study), use_container_width=True)
            with o2:
                st.plotly_chart(vis.plot_param_importances(study), use_container_width=True)
        except Exception:
            st.info("인터랙티브 차트를 보려면 `pip install plotly` 후 다시 실행하세요.")

    st.markdown("#### 🔍 SHAP 설명")
    st.caption("양(+) 값은 결제 쪽으로, 음(−) 값은 미결제 쪽으로 예측을 밀어내는 기여입니다.")

    explainer = shap.TreeExplainer(model)
    explanation = explainer(te)

    s1, s2, s3 = st.tabs(["전체 중요도 (bar)", "분포 (beeswarm)", "개별 예측"])
    with s1:
        st.write("어떤 변수가 결제 예측에 평균적으로 가장 큰 영향을 주는가")
        render(lambda: shap.plots.bar(explanation, show=False))
    with s2:
        st.write("변수 값(색)에 따라 결제/미결제 방향으로 어떻게 밀리는가")
        render(lambda: shap.plots.beeswarm(explanation, show=False))
    with s3:
        idx = st.slider("유저 인덱스", 0, len(te) - 1, 0)
        p = proba[idx]
        actual = CLASS_NAMES[int(y_test.iloc[idx])]
        pred_lab = CLASS_NAMES[int(p >= 0.5)]
        st.write(f"실제: **{actual}**  |  예측: **{pred_lab}** (결제확률 {p:.1%})")
        try:
            render(lambda: shap.plots.waterfall(explanation[idx], show=False))
        except Exception:
            render(lambda: shap.plots.bar(explanation[idx], show=False))
            st.caption("ⓘ 이 환경에서는 waterfall 렌더 이슈가 있어 bar로 표시했습니다.")
# ============================ Tab 3 ============================
with tab_use:
    st.subheader("💼 결제 전환 예측")
    st.caption(
        "체험이 끝난 고객의 이용 정보를 입력하면 **결제 전환 가능성**과 **메일 발송 추천 여부**를 알려줍니다. "
        "메일 프로모션 목적(놓치지 않기)에 맞춰 예측 모델은 **XGBoost**를 사용합니다."
    )

    # ── F2 1위 모델 안내 (모델 비교 결과를 예측 탭에서도 인지) ──
    # 현재 사이드바 설정 기준으로 5개 모델을 비교해 F2 1위를 확인하고,
    # 예측에 쓰는 XGBoost와 같은지/다른지 알려준다.
    _cmp_for_best, _ = compare_models(st.session_state["test_size"], st.session_state["balanced"])
    _best_row = _cmp_for_best.iloc[0]
    _best_name, _best_f2 = _best_row["모델"], _best_row["F2"]
    _xgb_f2 = float(_cmp_for_best.loc[_cmp_for_best["모델"] == "XGBoost", "F2"].iloc[0])
    if _best_name == "XGBoost":
        st.success(f"✅ 현재 설정에서 **F2 1위 모델이 XGBoost**(F2 {_best_f2:.3f})입니다. "
                   "예측에 쓰는 모델과 일치합니다.")
    else:
        st.info(f"ℹ️ 현재 설정에서 F2 1위는 **{_best_name}**(F2 {_best_f2:.3f})이고, "
                f"예측에는 **XGBoost**(F2 {_xgb_f2:.3f})를 사용합니다 — "
                "SHAP 해석·결측 자동처리 등 운영 일관성을 위해 XGBoost로 통일했습니다. "
                "최고 성능 모델을 바꾸려면 모델 비교 탭에서 지표를 확인하세요.")

    model_u, train_cols_u, explainer_u, scored_u = get_predictor(
        st.session_state["n_estimators"], st.session_state["max_depth"],
        st.session_state["learning_rate"], st.session_state["test_size"],
        st.session_state["balanced"],
    )
    base_rate = scored_u[TARGET].mean()

    mode_single, mode_batch, mode_track = st.tabs(
        ["🧑‍💼 고객 1명 예측", "📋 메일 발송 캠페인 타겟 리스트", "📈 성과 추적"])

    # ── 모드 A: 개별 고객 입력 → 예측 ──
    with mode_single:
        st.markdown("##### 고객 번호 입력 또는 옵션 직접 선택")
        src_s = st.radio("입력 방식", ["✍️ 직접 입력", "📂 데모·검증용 데이터", "📤 새 고객 파일 업로드"],
                         horizontal=True, key="single_src")
        prefill, cust_default = None, ""
        if "📂" in src_s:
            cid = st.selectbox("고객 선택 (고객번호)", scored_u["고객번호"].tolist(),
                               key="single_pick_builtin")
            prefill = scored_u.set_index("고객번호").loc[cid]
            cust_default = cid
            st.caption(f"참고 — 이 고객의 실제 결제 여부: "
                       f"**{'결제' if prefill[TARGET] == 1 else '미결제'}**")
        elif "📤" in src_s:
            up_s = st.file_uploader("고객 CSV 업로드 (모델 피처 포함)", type=["csv"], key="single_upload")
            if up_s is not None:
                dfu = pd.read_csv(io.BytesIO(up_s.getvalue()))
                miss = [c for c in [*NUM_PASS, *CAT_COLS] if c not in dfu.columns]
                if miss:
                    st.error("예측에 필요한 피처 컬럼이 없어요: " + ", ".join(miss))
                else:
                    ids = [c for c in dfu.columns
                           if c not in [*NUM_PASS, *CAT_COLS, TARGET, "pred_proba", "결제확률(%)"]]
                    if ids:
                        label = st.selectbox(f"고객 선택 ({ids[0]})",
                                             dfu[ids[0]].astype(str).tolist(), key="single_pick_file")
                        prefill = dfu[dfu[ids[0]].astype(str) == label].iloc[0]
                        cust_default = label
                    else:
                        ridx = st.selectbox("고객(행) 선택", range(len(dfu)),
                                            format_func=lambda i: f"{i + 1}행", key="single_pick_row")
                        prefill = dfu.iloc[ridx]
                        cust_default = f"{ridx + 1}행"

        def pv(col, default, lo=None, hi=None):
            """prefill에서 값 추출(없으면 default), 위젯 허용 범위로 clip."""
            v = prefill[col] if (prefill is not None and col in prefill
                                 and pd.notna(prefill[col])) else default
            if lo is not None:
                v = max(lo, v)
            if hi is not None:
                v = min(hi, v)
            return v

        site_val = int(pv("primary_site", SITE_IDS[0]))
        site_idx = SITE_IDS.index(site_val) if site_val in SITE_IDS else 0
        enter_default = (9 if int(pv("is_morning", 0)) == 1 else 14) if prefill is not None else 10

        with st.form("predict_one"):
            if prefill is not None:        # 데이터를 불러온 경우에만 식별자 표시(자동·읽기전용)
                cust = st.text_input("고객명 또는 고객 ID (불러온 값)", value=cust_default,
                                     disabled=True)
            else:                          # 직접 입력(빈 폼)에선 ID 칸을 두지 않음 — 예측에 쓰이지 않으므로
                cust = ""
            c1, c2, c3 = st.columns(3)
            with c1:
                dow = st.selectbox("가입 요일", list(range(7)), index=int(pv("signup_dow", 0, 0, 6)),
                                   format_func=lambda d: f"{DOW_KR[d]}요일")
                visit_days = st.radio("실제 방문일수 (체험 3일 중)", [1, 2, 3], horizontal=True,
                                      index=[1, 2, 3].index(int(pv("visit_days", 1, 1, 3))),
                                      help="체험 기간 동안 며칠 방문했는지")
                stay_hours = st.slider("총 체류시간 (시간)", 0.0, 50.0,
                                       round(float(pv("stay_total_sec", 28800, 0, 180000)) / 3600, 1), 0.5,
                                       help="체험 기간 전체 누적 체류시간")
                mean_hours = st.slider("방문 기록 1건당 평균 체류시간 (시간)", 0.0, 24.0,
                                       round(float(pv("stay_mean_sec", 14400, 0, 86400)) / 3600, 1), 0.5,
                                       help="한 번 방문 시 평균 머문 시간")
            with c2:
                n_access = st.number_input("출입 횟수 (게이트 태그)", 0, 300,
                                           int(pv("n_access_events", 8, 0, 300)))
                site = st.selectbox("주 이용 지점", SITE_IDS, index=site_idx,
                                    format_func=lambda s: f"{s}번 지점 ({SITE_AREA[s]}평)")
                n_sites = st.radio("방문 지점 수", [1, 2, 3, 4], horizontal=True,
                                   index=[1, 2, 3, 4].index(int(pv("n_sites_visited", 1, 1, 4))))
            with c3:
                gap = st.select_slider("마지막 방문 후 체험 종료까지 남은 일수", options=[0, 1, 2],
                                       value=int(pv("last_visit_to_end_gap", 0, 0, 2)),
                                       help="0 = 마지막 날까지 나옴(관심 ↑), 2 = 초반에만 나옴")
                enter_hour = st.slider("최초 입실 시간대 (시)", 0, 23, enter_default,
                                       help="가장 이른 방문의 입실 시각")
                st.caption(f"· 주 지점 면적: **{SITE_AREA[site]}평** (자동)")
                st.caption(f"· 입력한 시각이 12시 이전일 경우 **오전 이용**으로 반영"
                           if enter_hour < 12 else
                           "· 입력한 시각이 12시 이후일 경우 **오후 이용**으로 반영")
            submitted = st.form_submit_button("결제 가능성 예측하기", type="primary",
                                              use_container_width=True)

        if submitted:
            stay_total = stay_hours * 3600
            row = {
                "stay_total_sec": stay_total,
                "stay_mean_sec": mean_hours * 3600,            # 직원이 직접 입력
                "primary_site": float(site), "visit_days": float(visit_days),
                "n_access_events": float(n_access), "signup_dow": int(dow),
                "n_sites_visited": float(n_sites), "primary_site_area": float(SITE_AREA[site]),
                "last_visit_to_end_gap": float(gap),
                "is_morning": int(enter_hour < 12),            # 최초 입실 시각 → 오전 여부
            }
            X_one = build_input_row(row, train_cols_u)
            proba = float(model_u.predict_proba(X_one)[:, 1][0])

            who = cust.strip()
            st.markdown(f"##### ② 예측 결과 — {who}")
            r1, r2 = st.columns([1, 1.3])
            with r1:
                st.metric("결제 전환 확률", f"{proba:.1%}",
                          delta=f"평균 대비 {proba - base_rate:+.1%}p")
                st.progress(proba)
                thr = st.slider("메일 발송 기준(임계값)", 0.30, 0.70,
                                round(float(base_rate), 2), 0.01,
                                help="F2(놓치지 않기)를 중시하면 기준을 평균 전환율 근처로 낮춰 더 폭넓게 발송")
                if proba >= thr:
                    st.success("📧 **결제 유도 메일 발송 추천** — 전환 가능성이 기준점 이상입니다.")
                else:
                    st.warning("⏸️ **보류 권장** — 기준 미만. 비용 대비 효율이 낮을 수 있어요.")
            with r2:
                # SHAP 단일행 설명 → 한국어 자연어 top 기여
                expl = explainer_u(X_one)
                vals = expl.values[0]
                order = np.argsort(np.abs(vals))[::-1]
                st.markdown("**이 예측에 영향을 준 주요 요인**")
                shown = 0
                for i in order:
                    if abs(vals[i]) < 1e-6:
                        continue
                    arrow = "🔺 결제 가능성" if vals[i] > 0 else "🔻 미결제 가능성"
                    st.write(f"- {feature_kr(train_cols_u[i])} — {arrow} ({vals[i]:+.2f})")
                    shown += 1
                    if shown >= 4:
                        break
                st.caption("값은 결제 로그-오즈에 대한 기여도(SHAP)입니다.")

            # 비슷한 조건 고객의 실제 전환율 — 신뢰도 참고
            sim = scored_u[(scored_u["visit_days"] == visit_days)
                           & (scored_u["primary_site"] == site)]
            if len(sim) >= 10:
                st.info(f"참고: **{site}번 지점·{visit_days}일 방문** 고객 {len(sim):,}명의 "
                        f"실제 전환율은 **{sim[TARGET].mean():.1%}** 였습니다.")

    # ── 모드 B: 우선순위 타겟 리스트 (내장 데이터 or 업로드 파일) ──
    with mode_batch:
        st.markdown("##### 결제 가능성 높은 순으로 메일 발송 캠페인 대상 선정")
        src = st.radio("대상 고객 데이터", ["📂 데모·검증용 고객 (체험 종료)", "📤 새 파일 업로드 (CSV)"],
                       horizontal=True)
        settings = (st.session_state["n_estimators"], st.session_state["max_depth"],
                    st.session_state["learning_rate"], st.session_state["test_size"],
                    st.session_state["balanced"])

        scored, id_cols = None, []
        if "📂" in src:
            scored = scored_u
            id_cols = ["고객번호"]
            st.caption(f"체험 종료 고객 **{len(scored):,}명**을 결제확률 순으로 정렬합니다. "
                       "상위 그룹부터 공략하면 같은 메일 수로 더 많은 결제자를 잡습니다(타겟팅 효율). "
                       "📂 데모·검증용 — 결과를 이미 아는 과거 고객입니다. 모델이 실제로 맞았는지(Lift·적중률) 확인할 수 있어요.")

        else:
            st.caption("🚀 실제 운영용 — 결과를 모르는 신규 고객을 예측해 공략 대상을 정합니다.")
            up = st.file_uploader("고객 CSV 업로드", type=["csv"])
            with st.expander("업로드 파일에 필요한 컬럼 보기"):
                st.code(", ".join([*NUM_PASS, *CAT_COLS]))
                st.caption("그 외 컬럼(user_uuid·고객명·이메일 등)은 결과에 그대로 보존됩니다. "
                           "`is_payment`가 있으면 적중률·Lift까지 계산합니다. "
                           "또는 `이메일·고객명·결제확률·제목·본문`이 들어있는 메일 발송용 CSV도 올릴 수 있어요.")
            if up is not None:
                file_bytes = up.getvalue()
                scored, missing, id_cols = predict_uploaded(file_bytes, settings)
                if missing:
                    prepared = prepare_prescored_mail_file(file_bytes)
                    if prepared is not None:
                        scored, id_cols = prepared
                        missing = []
                        st.success("메일 발송용 CSV로 인식했습니다. 기존 결제확률·제목·본문을 사용합니다.")
                    else:
                        st.error("이 파일로는 예측할 수 없어요. 모델이 쓰는 영문 피처 컬럼이 없습니다: "
                                 + ", ".join(missing))
                        st.info("· **신규 고객 예측**이라면: 위 영문 피처가 담긴 데이터 파일을 올려주세요.\n\n"
                                "· **이미 만든 메일 발송 명단**이라면: `이메일`, `결제확률`, `제목`, `본문` 컬럼이 필요합니다.\n\n"
                                "· **캠페인 결과 분석**이라면: 이 탭이 아니라 **📈 성과 추적** 탭에 올려주세요 "
                                "(거긴 `실제결제여부`만 있으면 됩니다).")
                        scored = None
            else:
                st.info("CSV를 올리면 자동으로 결제확률을 계산해 우선순위를 매깁니다.")

        if scored is not None:
            top_pct = st.slider("상위 몇 %에게 발송할까요?", 5, 100, 30, 5)
            n_target = max(1, int(len(scored) * top_pct / 100))
            target = scored.sort_values("pred_proba", ascending=False).head(n_target).copy()
            has_target = TARGET in scored.columns

            cols = st.columns(3)
            cols[0].metric("발송 대상", f"{n_target:,}명", f"전체의 {top_pct}%")
            if has_target:
                captured, total_p = target[TARGET].sum(), scored[TARGET].sum()
                base = scored[TARGET].mean()
                cols[1].metric("잡히는 결제자", f"{int(captured):,}명",
                               f"전체 결제자의 {captured / total_p:.0%}")
                cols[2].metric("타겟팅 효율(Lift)", f"{target[TARGET].mean() / base:.2f}배",
                               help="무작위 발송 대비 결제자 밀도")
            else:
                cols[1].metric("평균 결제확률", f"{target['pred_proba'].mean():.1%}")
                cols[2].metric("최고 결제확률", f"{target['pred_proba'].max():.1%}")

            # ── 캠페인 KPI 카드 (실제 결제여부를 아는 데모 데이터일 때) ──
            # "상위 N%에게 메일 발송"을 하나의 캠페인으로 보고, 그 성과를 4개 지표로 풀어 설명
            if has_target:
                st.divider()
                st.markdown(f"##### 📨 이 캠페인의 예상 성과 (상위 {top_pct}%에게 발송 시)")

                # 발송(상위 N%)을 '결제 예측=1', 나머지를 '0'으로 본 혼동행렬 개념
                sent = target                                   # 메일 보낼 대상
                not_sent = scored.sort_values("pred_proba", ascending=False).iloc[n_target:]
                tp = int(sent[TARGET].sum())                    # 보냈고 실제 결제
                fp = int((sent[TARGET] == 0).sum())             # 보냈는데 미결제(헛메일)
                fn = int(not_sent[TARGET].sum())                # 안 보냈는데 실제 결제(놓침)
                actual_positive = tp + fn                       # 전체 실제 결제자
                sent_total = tp + fp                            # 메일 발송 총수
                detect = tp / actual_positive if actual_positive else 0   # 결제자 탐지율
                mail_acc = tp / sent_total if sent_total else 0           # 메일 정확도

                k1, k2, k3, k4 = st.columns(4)
                # 카드 하단 설명(caption)은 제거 — metric 수치만 노출
                with k1:
                    st.metric("🎯 결제자 탐지율", f"{detect:.1%}")
                with k2:
                    st.metric("📬 메일 정확도", f"{mail_acc:.1%}")
                with k3:
                    st.metric("📮 메일 발송 대상", f"{sent_total:,}명")
                with k4:
                    st.metric("😥 놓치는 결제자", f"{fn:,}명")

                st.info(f"💡 상위 **{top_pct}%**에게만 보내도 전체 결제자의 "
                        f"**{detect:.0%}**를 잡습니다. 발송 비율을 높이면 탐지율은 오르지만 "
                        f"메일 낭비(헛메일)도 함께 늘어요 — 메일 비용에 맞춰 조절하세요.")

            target["결제확률(%)"] = (target["pred_proba"] * 100).round(1)
            # 화면 표 (한글 요약, 보기용)
            keep = id_cols + ["결제확률(%)", "primary_site", "visit_days", "stay_total_sec",
                              "n_access_events", "last_visit_to_end_gap"] + ([TARGET] if has_target else [])
            view = target[[c for c in keep if c in target.columns]].copy()
            if "stay_total_sec" in view:
                view["stay_total_sec"] = (view["stay_total_sec"] / 3600).round(1)
            view = view.rename(columns={
                "primary_site": "주지점", "visit_days": "방문일수", "stay_total_sec": "총체류(시간)",
                "n_access_events": "출입수", "last_visit_to_end_gap": "종료간격", TARGET: "실제결제",
            })
            st.dataframe(view, use_container_width=True, hide_index=True, height=360)

            # ─────────────────────────────────────────────────────────────
            # 메일 발송 액션 영역
            # ─────────────────────────────────────────────────────────────
            st.divider()
            st.markdown("##### ✉️ 타겟 고객 메일 발송")
            st.caption(
                "위에서 추출한 상위 타겟 고객에게 바로 메일을 보낼 수 있습니다. "
                "실제 운영 전에는 반드시 테스트 메일로 먼저 확인하세요."
            )

            email_ready = "email" in target.columns
            if not email_ready:
                st.warning(
                    "현재 타겟 데이터에 `email` 컬럼이 없습니다. "
                    "실제 고객 발송을 하려면 업로드 CSV에 `email` 컬럼을 추가하세요. "
                    "발표 데모는 아래 테스트 수신 이메일로 진행할 수 있습니다."
                )

            m1, m2 = st.columns([1, 1])
            with m1:
                mail_subject = st.text_input(
                    "메일 제목",
                    "[공유오피스] 3일 체험 고객님을 위한 첫 결제 혜택 안내",
                    key="campaign_mail_subject",
                )
            with m2:
                benefit_text = st.text_input(
                    "추천 혜택 문구",
                    "첫 결제 고객 대상 10% 할인 쿠폰",
                    key="campaign_benefit_text",
                )

            send_mode = st.radio(
                "발송 방식",
                ["테스트 발송", "실제 고객 발송"],
                horizontal=True,
                help="테스트 발송은 선택한 1명의 메일 내용을 입력한 테스트 주소로만 보냅니다.",
                key="campaign_send_mode",
            )

            preview_options = target.index.tolist()
            preview_idx = st.selectbox(
                "메일 미리보기 고객",
                preview_options,
                format_func=lambda i: str(target.loc[i].get("email", target.loc[i].get("고객번호", i))),
                key="campaign_preview_customer",
            )
            preview_body = build_campaign_email_body(target.loc[preview_idx], benefit_text)
            with st.expander("메일 본문 미리보기", expanded=True):
                st.write(f"**제목:** {get_campaign_subject(target.loc[preview_idx], mail_subject)}")
                st.markdown(preview_body, unsafe_allow_html=True)

            if send_mode == "테스트 발송":
                test_email = st.text_input("테스트 수신 이메일", key="campaign_test_email")
                can_send = bool(test_email)
                if st.button("테스트 메일 발송", type="primary", disabled=not can_send, use_container_width=True):
                    try:
                        send_campaign_email(
                            test_email,
                            get_campaign_subject(target.loc[preview_idx], mail_subject),
                            preview_body,
                        )
                        append_email_log([{
                            "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "mode": "test",
                            "to_email": test_email,
                            "customer": str(target.loc[preview_idx].get("고객번호", preview_idx)),
                            "pred_proba": float(target.loc[preview_idx].get("pred_proba", 0)),
                            "status": "success",
                            "error": "",
                        }])
                        st.success("테스트 메일을 발송했습니다.")
                    except Exception as e:
                        st.error(f"테스트 메일 발송 실패: {e}")
            else:
                if not email_ready:
                    st.info("실제 고객 발송은 `email` 컬럼이 있는 CSV를 업로드했을 때 활성화됩니다.")
                else:
                    send_limit = st.number_input(
                        "최대 발송 건수",
                        min_value=1,
                        max_value=len(target),
                        value=min(10, len(target)),
                        step=1,
                        help="실수 방지를 위해 한 번에 보낼 최대 건수를 제한합니다.",
                        key="campaign_send_limit",
                    )
                    confirm_send = st.checkbox(
                        "실제 고객에게 메일을 발송하는 것을 확인했습니다.",
                        key="campaign_confirm_send",
                    )
                    if st.button("실제 고객 메일 발송", type="primary", disabled=not confirm_send, use_container_width=True):
                        send_df = target.head(int(send_limit)).copy()
                        success_count, fail_count = 0, 0
                        log_rows = []

                        progress = st.progress(0)
                        for i, (_, row) in enumerate(send_df.iterrows(), start=1):
                            to_email = str(row.get("email", "")).strip()
                            row_subject = get_campaign_subject(row, mail_subject)
                            body = build_campaign_email_body(row, benefit_text)
                            log = {
                                "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "mode": "real",
                                "to_email": to_email,
                                "customer": str(row.get("고객번호", row.name)),
                                "pred_proba": float(row.get("pred_proba", 0)),
                                "status": "",
                                "error": "",
                            }
                            try:
                                if not to_email:
                                    raise ValueError("email 값이 비어 있습니다.")
                                send_campaign_email(to_email, row_subject, body)
                                success_count += 1
                                log["status"] = "success"
                            except Exception as e:
                                fail_count += 1
                                log["status"] = "fail"
                                log["error"] = str(e)
                            log_rows.append(log)
                            progress.progress(i / len(send_df))

                        append_email_log(log_rows)
                        st.success(f"메일 발송 완료: 성공 {success_count}건 / 실패 {fail_count}건")
                        if fail_count:
                            st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

            # 다운로드 (원본 영문 피처 전부 보존 → 그대로 재예측·성과추적 가능, 추적 빈 칸 동봉)
            dl = target.drop(columns=["pred_proba"]).copy()
            for tc in TRACK_COLS:
                dl[tc] = ""
            st.download_button(
                "📥 발송 대상 명단 CSV 다운로드 (성과 추적용 빈 칸 포함)",
                dl.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"메일발송대상자_상위{top_pct}%_{n_target}명.csv",
                mime="text/csv", use_container_width=True,
            )
            st.caption("💡 받은 CSV의 빈 칸(연락여부·실제결제여부 등)을 캠페인 후 채워서 "
                       "**📈 성과 추적** 탭에 올리면 전략 효과를 분석합니다.")

    # ── 모드 C: 성과 추적 (캠페인 결과 업로드 → 효과 분석) ──
    with mode_track:
        st.markdown("##### 캠페인 결과를 올려 전략 효과를 확인하세요")
        st.caption("‘메일 캠페인 타겟 리스트’에서 받은 CSV에 결과를 채워 올리면 "
                   "**예측 → 연락 → 결제 결과 기록 → 효과 확인** 루프가 닫힙니다.")
        with st.expander("채워야 할 컬럼 보기"):
            st.code(", ".join(TRACK_COLS))
            st.caption("연락여부·쿠폰제공여부·실제결제여부는 예/아니오(또는 Y/N, 1/0)로 입력하세요. "
                       "모델 점수 구간 분석에는 ‘결제확률(%)’ 컬럼이 필요합니다(타겟 리스트에 이미 포함).")
        up_t = st.file_uploader("캠페인 결과 CSV 업로드", type=["csv"], key="track_upload")

        if up_t is None:
            st.info("결과 파일을 올리면 ▸ 우선 상담 결제율 ▸ 연락 효과 ▸ 안내 방식별 "
                    "▸ 모델 점수 구간별 결제율을 분석합니다.")
        else:
            df_t = pd.read_csv(io.BytesIO(up_t.getvalue()))
            if "실제결제여부" not in df_t.columns:
                st.error("'실제결제여부' 컬럼이 필요합니다. 타겟 리스트에서 받은 양식을 채워 올려주세요.")
            else:
                paid = df_t["실제결제여부"].map(to_yes)

                st.markdown("##### ① 전체 성과")
                k1, k2, k3 = st.columns(3)
                k1.metric("기록된 고객", f"{len(df_t):,}명")
                k2.metric("실제 결제", f"{int(paid.sum()):,}명")
                k3.metric("전체 결제율", f"{paid.mean():.1%}")

                # ② 연락 효과 (우선 상담 그룹 결제율 포함)
                if "연락여부" in df_t.columns and df_t["연락여부"].astype(str).str.strip().replace("nan", "").any():
                    con = df_t["연락여부"].map(to_yes)
                    st.markdown("##### ② 연락 효과 —  컨택 고객 vs 미컨택 고객")
                    r_con = paid[con].mean() if con.any() else 0.0
                    r_non = paid[~con].mean() if (~con).any() else 0.0
                    cc1, cc2 = st.columns(2)
                    cc1.metric("컨택 고객 결제율", f"{r_con:.1%}")
                    cc2.metric("미컨택 고객 결제율", f"{r_non:.1%}", delta=f"{r_con - r_non:+.1%}p")
                    st.bar_chart(paid.groupby(con.map({True: "컨택", False: "미컨택"}))
                                 .mean().mul(100).round(1))
                    st.caption("막대는 그룹별 실제 결제율(%). 컨택한 그룹이 더 높을수록 전환에 기여한 것입니다.")
                else:
                    st.markdown("##### ② 연락 효과")
                    st.caption("‘연락여부’가 아직 비어 있어요. 캠페인 후 예/아니오를 채워 올리면 "
                               "연락한 고객과 안 한 고객의 결제율을 비교해 드립니다.")

                # ③ 안내 방식별 결제율 (재방문 안내 등)
                if "안내방식" in df_t.columns and df_t["안내방식"].notna().any():
                    st.markdown("##### ③ 안내 방식별 결제율")
                    method = df_t["안내방식"].fillna("(미입력)").replace("", "(미입력)")
                    by_m = paid.groupby(method).agg(["mean", "size"])
                    st.bar_chart((by_m["mean"] * 100).round(1))
                    st.caption("‘재방문 안내’ 등 안내 방식별 결제 전환율(%). 다음 캠페인 진행 시 참고하세요.")

                # ④ 쿠폰 제공 효과
                if "쿠폰제공여부" in df_t.columns:
                    cp = df_t["쿠폰제공여부"].map(to_yes)
                    if cp.any() and (~cp).any():
                        st.markdown("##### ④ 쿠폰 제공 효과")
                        d1, d2 = st.columns(2)
                        d1.metric("쿠폰 제공 그룹 결제율", f"{paid[cp].mean():.1%}")
                        d2.metric("쿠폰 미제공 그룹 결제율", f"{paid[~cp].mean():.1%}",
                                  delta=f"{paid[cp].mean() - paid[~cp].mean():+.1%}p")

                # ⑤ 모델 점수 구간별 실제 결제율 (캘리브레이션 검증)
                score_col = ("결제확률(%)" if "결제확률(%)" in df_t.columns
                             else "pred_proba" if "pred_proba" in df_t.columns else None)
                if score_col:
                    st.markdown("##### ⑤ 모델 점수 구간별 실제 결제율")
                    score = df_t[score_col].astype(float)
                    if score.max() <= 1.5:        # 0~1 스케일이면 %로 환산
                        score = score * 100
                    seg = pd.cut(score, [0, 20, 40, 60, 80, 100],
                                 labels=["0-20", "20-40", "40-60", "60-80", "80-100"],
                                 include_lowest=True)
                    st.bar_chart((paid.groupby(seg, observed=False).mean() * 100).round(1))
                    st.caption("점수가 높은 구간일수록 실제 결제율도 높으면(우상향) 모델이 잘 맞는 것입니다.")

                st.success("✅ 루프 완료 — 효과가 좋았던 안내 방식·점수 구간에 다음 캠페인을 집중하세요.")


# ============================ Tab 4 ============================
# 메일 CSV 생성 — 부록참고 노트북의 발송파일을 대시보드에서 생성 → n8n이 이 CSV를 먹음
with tab_mail:
    st.subheader("📧 메일 발송용 CSV 생성 (n8n 자동화 입력)")
    st.caption(
        "고객 CSV를 올리면 **왼쪽 사이드바에서 최적화한 모델(옵튜나/수동 하이퍼파라미터)**로 채점하고, "
        "**미결제 + 체험종료** 고객 중 **상위 N%**를 골라 세그먼트·제목·본문까지 붙인 "
        "df_master_mail_ver2.csv 양식으로 만들어 줍니다. n8n 워크플로우가 이 CSV를 그대로 읽어 발송합니다."
    )

    # 현재 사이드바 모델 설정 표시 (옵튜나 지표 포함)
    _opt = st.session_state.get("opt_metric")
    _cfg = (f"n_estimators={st.session_state['n_estimators']}, "
            f"max_depth={st.session_state['max_depth']}, "
            f"learning_rate={st.session_state['learning_rate']}, "
            f"불균형보정={'ON' if st.session_state['balanced'] else 'OFF'}")
    st.info(f"🤖 채점에 쓸 현재 모델: {_cfg}"
            + (f"  ·  Optuna 최적화 지표: **{_opt}**" if _opt else "  ·  (사이드바에서 옵튜나/수동 조정 가능)"))

    up_m = st.file_uploader("고객 CSV 업로드 (user_uuid·trial_end·is_payment·모델 피처 포함)",
                            type=["csv"], key="mail_upload")

    with st.expander("📋 입력 CSV 컬럼 안내 & 이메일 로직 (누르면 펼쳐짐)"):
        st.markdown(
            "**필수 컬럼** (없으면 생성 안 됨)\n"
            "- 모델 피처 10개: `stay_total_sec`, `stay_mean_sec`, `visit_days`, `n_access_events`, "
            "`n_sites_visited`, `primary_site_area`, `last_visit_to_end_gap`, `is_morning`, "
            "`primary_site`, `signup_dow`\n"
            "- `is_payment` (결제 여부 · 미결제자만 발송 대상으로 거름)\n"
            "- `user_uuid` (고객 식별 · 플러스 어드레싱 태그로도 사용)\n"
            "- `trial_end` (체험 종료일 · '체험 끝난 고객'만 거를 때 사용)\n\n"
            "**선택 컬럼** (있으면 활용, 없으면 자동 처리)\n"
            "- `email` (실제 이메일 · 아래 입력창 로직 참고)\n"
            "- `trial_date` (없으면 빈칸)\n"
            "- `last_visit_date` (없으면 빈칸)\n"
            "- `trial_key` (없으면 `user_uuid_trial_end`로 자동 생성 · n8n 중복발송 방지 키)\n\n"
            "**이메일 입력창 로직 (우선순위)**\n"
            "1. 입력창을 **채우면** → 모든 고객을 `입력이메일+{user_uuid}@도메인` 으로 생성 "
            "(`email` 컬럼 유무와 무관, `@` 없이 아이디만 넣으면 `@gmail.com` 자동)\n"
            "2. 입력창 **비우고** CSV에 `email` 컬럼 있으면 → 그 `email`을 **그대로** 사용\n"
            "3. 입력창 **비우고** `email` 컬럼도 없으면 → 데모 이메일(`test+0001@example.com`)"
        )

    cc = st.columns(2)
    base_email_m = cc[0].text_input(
        "기본 이메일 (플러스 어드레싱용)", value="",
        placeholder="예: abcd1234@gmail.com  또는  abcd1234",
        help="채우면 각 고객을 이메일+{user_uuid} 로 만들어 본인 함으로 모읍니다. "
             "비우면 CSV의 email 컬럼을 그대로 쓰고(있을 때), 그것도 없으면 데모 이메일을 씁니다.")
    file_name_m = cc[1].text_input(
        "다운로드 파일명", value="df_master_mail",
        help="확장자(.csv)는 자동으로 붙습니다.")

    top_pct_m = st.slider("미결제 방문자 중 상위 몇 %에게 발송할까요?", 5, 100, 10, 5,
                          help="부록참고 노트북 기본값은 상위 10%입니다.")

    if up_m is None:
        st.info("CSV를 업로드하면 발송 대상 파일이 생성됩니다. "
                "예: df_master_with_date.csv (user_uuid, trial_end, is_payment, 피처 컬럼 포함)")
    else:
        result = None
        try:
            result, info = build_mail_table(
                up_m.getvalue(), top_pct_m,
                st.session_state["n_estimators"], st.session_state["max_depth"],
                st.session_state["learning_rate"], st.session_state["test_size"],
                st.session_state["balanced"], base_email_m,
            )
        except Exception as e:
            st.error(f"생성 실패: {e}")

        if result is not None and len(result):
            c = st.columns(3)
            c[0].metric("미결제·체험종료 후보", f"{info['eligible']:,}명")
            c[1].metric(f"발송 대상 (상위 {top_pct_m}%)", f"{info['n_mail']:,}명")
            c[2].metric("세그먼트(높음/중간)",
                        f"{info['seg'].get('높음', 0)} / {info['seg'].get('중간', 0)}")

            if not info["all_unpaid"]:
                st.warning("⚠️ 결제 완료 고객이 섞여 있습니다. 입력 CSV의 is_payment를 확인하세요.")
            if not info["key_unique"]:
                st.warning("⚠️ trial_key가 중복됩니다. n8n 중복 발송 방지가 약해질 수 있어요.")

            st.markdown("##### 미리보기 (앞 20행)")
            st.dataframe(result.head(20), use_container_width=True, hide_index=True)

            _fn = (file_name_m or "df_master_mail_ver2").strip()
            if not _fn.lower().endswith(".csv"):
                _fn += ".csv"
            st.download_button(
                f"⬇️ {_fn} 다운로드",
                result.to_csv(index=False).encode("utf-8-sig"),
                file_name=_fn, mime="text/csv",
                type="primary", use_container_width=True,
            )
            st.caption("이 CSV의 컬럼(이메일·세그먼트·제목·본문 등)은 n8n 워크플로우 "
                       "(세그먼트=높음 필터 → 메일 발송)에 그대로 맞춰져 있습니다.")
        elif result is not None:
            st.warning("발송 대상이 0명입니다. 상위 % 를 높이거나 입력 데이터를 확인하세요.")


st.divider()
st.caption("공유오피스 전환 예측 · 5-model · 모델별 전처리 · XGBoost · Optuna(F2/정확도/Recall) · SHAP")
