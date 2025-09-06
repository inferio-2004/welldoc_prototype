# backend/app.py
# Full FastAPI backend with disease-specific reports added.
import os
import io
import json
import base64
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import concurrent.futures
import textwrap
import difflib
app = FastAPI(title="Patient Predictor API (Disease reports)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
MODEL_PATH = os.environ.get("MODEL_PATH", "logistic_model.pkl")
FEATURES_TXT = os.environ.get("FEATURES_TXT", "derived_cols.txt")
LLM_GGUF_PATH = os.environ.get("LLM_GGUF_PATH", "models\gemma-2b.Q4_K_M.gguf")
AUTO_RUN_LLM = os.environ.get("AUTO_RUN_LLM", "true").lower() in ("1", "true", "yes")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "200"))

# Load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Warning: could not load model at {MODEL_PATH}: {e}")

# Try to load llama_cpp Llama
try:
    from llama_cpp import Llama
    try:
        llm = Llama(model_path=LLM_GGUF_PATH, temperature=0.2,verbose=False)
        print("Loaded offline GGUF LLM at", LLM_GGUF_PATH)
    except Exception as e:
        print("Failed to init Llama:", e)
        llm = None
except Exception as e:
    llm = None
    print("llama_cpp not available:", e)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# ---------- helpers ----------
def sanitize_colname(c: str) -> str:
    return (
        c.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(",", "")
    )

def parse_yes_no(value: Any) -> int:
    if pd.isna(value):
        return 0
    s = str(value).strip().lower()
    if s in ("yes", "y", "true", "1", "t", "male", "m"):
        return 1
    return 0

def to_float_safe(x, default=np.nan):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def read_features_txt(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
    lines = [ln for ln in lines if 'deteriorated' not in ln.lower() and 'target' not in ln.lower()]
    return [sanitize_colname(x) for x in lines]

FEATURE_NAMES = read_features_txt(FEATURES_TXT)
if FEATURE_NAMES:
    print(f"Using derived features from {FEATURES_TXT}: {len(FEATURE_NAMES)}")
else:
    if model is not None and hasattr(model, "feature_names_in_"):
        FEATURE_NAMES = [sanitize_colname(x) for x in list(model.feature_names_in_)]
        print("Using model.feature_names_in_ (sanitized).")
    else:
        print("No derived features file and model.feature_names_in_ not found. Will infer features from CSV.")

def compute_slope(series: pd.Series) -> float:
    y = series.dropna().astype(float).values
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n)
    try:
        m, _ = np.polyfit(x, y, 1)
        return float(m)
    except Exception:
        return float((y[-1] - y[0]) / (n - 1))

def find_col_like(df, patterns: List[str]):
    for c in df.columns:
        lc = c.lower()
        for p in patterns:
            if p in lc:
                return c
    return None

def pretty_feature_name(feat: str) -> str:
    s = feat.replace("_", " ").strip()
    s = s.replace("avg_7d", "average 7 days").replace("avg 7d", "average 7 days").replace("avg", "average")
    s = s.replace("slope_7d", "slope per day").replace("volatility_7d", "volatility 7 days")
    s = s.replace("min_7d", "min 7 days")
    s = s.replace("hr", "heart rate").replace("spo2", "SpO2").replace("bp", "blood pressure")
    s = s.replace("pi", "pollution index")
    s = " ".join(s.split())
    s = s.replace("cholestrol", "cholesterol")  # fix common typo
    return s

# compute derived features (same as before)
def compute_derived_from_timeseries(df_patient: pd.DataFrame) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if 'day' in df_patient.columns:
        df_patient = df_patient.sort_values('day')
    last7 = df_patient.tail(7)

    col_chol = find_col_like(df_patient, ["cholesterol", "chol"])
    col_glucose = find_col_like(df_patient, ["glucose", "sugar"])
    col_bp = find_col_like(df_patient, ["systolic", "blood pressure", "bp"])
    col_weight = find_col_like(df_patient, ["weight", "wt"])
    col_spo2 = find_col_like(df_patient, ["spo2", "sp02", "o2", "saturation"])
    col_hr = find_col_like(df_patient, ["heart_rate", "heart rate", "hr", "pulse"])
    col_sleep = find_col_like(df_patient, ["sleep", "sleep_hours", "sleep_time"])
    col_bmi = find_col_like(df_patient, ["bmi"])
    col_pollution = find_col_like(df_patient, ["pollution", "pi", "pollution_index"])
    col_smoker = find_col_like(df_patient, ["smoker", "smoking"])
    col_asthma = find_col_like(df_patient, ["asthma"])
    col_diabetes = find_col_like(df_patient, ["diabetes"])
    col_hd = find_col_like(df_patient, ["heart disease", "heart_disease", "heart"])
    col_family = find_col_like(df_patient, ["family", "family_history", "family history"])
    col_wheezing = find_col_like(df_patient, ["wheezing", "wheeze"])
    col_sex = find_col_like(df_patient, ["sex", "gender"])
    col_chest = find_col_like(df_patient, ["chest", "chest pain", "chest_pain"])
    col_age = find_col_like(df_patient, ["age"])

    def safe_last(col):
        if col is None or col not in df_patient.columns:
            return np.nan
        vals = pd.to_numeric(df_patient[col].dropna(), errors='coerce')
        return float(vals.values[-1]) if vals.size > 0 else np.nan

    chol_vals = pd.to_numeric(last7[col_chol], errors='coerce') if col_chol else pd.Series(dtype=float)
    chol_latest = safe_last(col_chol)
    out["cholesterol_latest"] = float(np.nan_to_num(chol_latest, nan=0.0))
    out["cholesterol_avg_7d"] = float(np.nan_to_num(chol_vals.mean(), nan=0.0))
    out["cholesterol_slope_7d"] = float(compute_slope(chol_vals)) if len(chol_vals) > 0 else 0.0
    out["cholesterol_volatility_7d"] = float(np.nan_to_num(chol_vals.std(ddof=0), nan=0.0))
    out["cholesterol_days_above_200_last7"] = float(np.sum(chol_vals > 200)) if len(chol_vals) > 0 else 0.0

    gl_vals = pd.to_numeric(last7[col_glucose], errors='coerce') if col_glucose else pd.Series(dtype=float)
    gl_latest = safe_last(col_glucose)
    out["glucose_latest"] = float(np.nan_to_num(gl_latest, nan=0.0))
    out["glucose_avg_7d"] = float(np.nan_to_num(gl_vals.mean(), nan=0.0))
    out["glucose_slope_7d"] = float(compute_slope(gl_vals)) if len(gl_vals) > 0 else 0.0
    out["glucose_volatility_7d"] = float(np.nan_to_num(gl_vals.std(ddof=0), nan=0.0))
    out["glucose_days_above_180_last7"] = float(np.sum(gl_vals > 180)) if len(gl_vals) > 0 else 0.0

    bp_vals = pd.to_numeric(last7[col_bp], errors='coerce') if col_bp else pd.Series(dtype=float)
    bp_latest = safe_last(col_bp)
    out["bp_latest"] = float(np.nan_to_num(bp_latest, nan=0.0))
    out["bp_avg_7d"] = float(np.nan_to_num(bp_vals.mean(), nan=0.0))
    out["bp_slope_7d"] = float(compute_slope(bp_vals)) if len(bp_vals) > 0 else 0.0
    out["bp_volatility_7d"] = float(np.nan_to_num(bp_vals.std(ddof=0), nan=0.0))
    out["bp_days_above_140_last7"] = float(np.sum(bp_vals > 140)) if len(bp_vals) > 0 else 0.0

    wt_vals = pd.to_numeric(last7[col_weight], errors='coerce') if col_weight else pd.Series(dtype=float)
    wt_latest = safe_last(col_weight)
    out["weight_latest"] = float(np.nan_to_num(wt_latest, nan=0.0))
    out["weight_avg_7d"] = float(np.nan_to_num(wt_vals.mean(), nan=0.0))
    out["weight_volatility_7d"] = float(np.nan_to_num(wt_vals.std(ddof=0), nan=0.0))

    spo2_vals = pd.to_numeric(last7[col_spo2], errors='coerce') if col_spo2 else pd.Series(dtype=float)
    spo2_latest = safe_last(col_spo2)
    out["spo2_latest"] = float(np.nan_to_num(spo2_latest, nan=0.0))
    out["spo2_avg_7d"] = float(np.nan_to_num(spo2_vals.mean(), nan=0.0))
    out["spo2_min_7d"] = float(np.nan_to_num(spo2_vals.min(), nan=0.0)) if len(spo2_vals) > 0 else 0.0
    out["spo2_slope_7d"] = float(compute_slope(spo2_vals)) if len(spo2_vals) > 0 else 0.0
    out["spo2_days_below_92_last7"] = float(np.sum(spo2_vals < 92)) if len(spo2_vals) > 0 else 0.0

    hr_vals = pd.to_numeric(last7[col_hr], errors='coerce') if col_hr else pd.Series(dtype=float)
    hr_latest = safe_last(col_hr)
    out["hr_latest"] = float(np.nan_to_num(hr_latest, nan=0.0))
    out["hr_avg_7d"] = float(np.nan_to_num(hr_vals.mean(), nan=0.0))
    out["hr_slope_7d"] = float(compute_slope(hr_vals)) if len(hr_vals) > 0 else 0.0
    out["hr_volatility_7d"] = float(np.nan_to_num(hr_vals.std(ddof=0), nan=0.0))
    out["hr_days_above_100_last7"] = float(np.sum(hr_vals > 100)) if len(hr_vals) > 0 else 0.0

    sleep_vals = pd.to_numeric(last7[col_sleep], errors='coerce') if col_sleep else pd.Series(dtype=float)
    sleep_latest = safe_last(col_sleep)
    out["sleep_latest"] = float(np.nan_to_num(sleep_latest, nan=0.0))
    out["sleep_avg_7d"] = float(np.nan_to_num(sleep_vals.mean(), nan=0.0))
    out["sleep_slope_7d"] = float(compute_slope(sleep_vals)) if len(sleep_vals) > 0 else 0.0
    out["sleep_volatility_7d"] = float(np.nan_to_num(sleep_vals.std(ddof=0), nan=0.0))
    out["sleep_days_below_6h_last7"] = float(np.sum(sleep_vals < 6)) if len(sleep_vals) > 0 else 0.0

    poll_vals = pd.to_numeric(last7[col_pollution], errors='coerce') if col_pollution else pd.Series(dtype=float)
    poll_latest = safe_last(col_pollution)
    out["pollution_index_latest"] = float(np.nan_to_num(poll_latest, nan=0.0))

    if col_wheezing and df_patient[col_wheezing].dropna().size>0:
        out["wheezing_problem"] = float(parse_yes_no(df_patient[col_wheezing].dropna().iloc[-1]))
    else:
        out["wheezing_problem"] = 0.0

    out["bmi"] = float(np.nan_to_num(safe_last(col_bmi), nan=0.0))

    out["smoker"] = float(parse_yes_no(df_patient[col_smoker].dropna().iloc[-1])) if (col_smoker and df_patient[col_smoker].dropna().size>0) else 0.0
    out["heart_disease"] = float(parse_yes_no(df_patient[col_hd].dropna().iloc[-1])) if (col_hd and df_patient[col_hd].dropna().size>0) else 0.0
    out["diabetes"] = float(parse_yes_no(df_patient[col_diabetes].dropna().iloc[-1])) if (col_diabetes and df_patient[col_diabetes].dropna().size>0) else 0.0
    out["asthma"] = float(parse_yes_no(df_patient[col_asthma].dropna().iloc[-1])) if (col_asthma and df_patient[col_asthma].dropna().size>0) else 0.0
    out["family_history_asthma"] = float(parse_yes_no(df_patient[col_family].dropna().iloc[-1])) if (col_family and df_patient[col_family].dropna().size>0) else 0.0

    out["sex_m"] = 1.0 if (col_sex and str(df_patient[col_sex].dropna().iloc[-1]).strip().lower() in ("m", "male", "1")) else 0.0
    out["chest_pain_type"] = float(to_float_safe(df_patient[col_chest].dropna().iloc[-1])) if (col_chest and df_patient[col_chest].dropna().size>0) else 0.0

    out["age"] = float(to_float_safe(df_patient[col_age].dropna().iloc[-1])) if (col_age and df_patient[col_age].dropna().size>0) else 0.0

    if FEATURE_NAMES:
        for f in FEATURE_NAMES:
            if f not in out:
                out[f] = 0.0

    return out

# Align & contributions helpers
def _san_key(s: str) -> str:
    return sanitize_colname(str(s))

def align_X_to_model(Xpd: pd.DataFrame) -> pd.DataFrame:
    if model is None or not hasattr(model, "feature_names_in_"):
        return Xpd
    model_names = list(model.feature_names_in_)
    model_map = { _san_key(n): n for n in model_names }
    col_map = {}
    for c in Xpd.columns:
        k = _san_key(c)
        if k in model_map:
            col_map[c] = model_map[k]
        else:
            col_map[c] = c
    X_aligned = Xpd.rename(columns=col_map)
    X_aligned = X_aligned.reindex(columns=model_names, fill_value=0.0)
    return X_aligned

def compute_top3_contributions_from_vector_with_pct(coef: np.ndarray, feature_list: List[str], x_vector: np.ndarray):
    arr_coef = coef.reshape(-1)
    contrib = arr_coef * x_vector.reshape(-1)
    df = pd.DataFrame({
        "feature": feature_list,
        "coefficient": arr_coef,
        "value": x_vector.reshape(-1),
        "contribution": contrib
    })
    df["abs_contrib"] = df["contribution"].abs()
    total_abs = df["abs_contrib"].sum()
    if total_abs == 0:
        df["contribution_pct"] = 0.0
        df["abs_pct"] = 0.0
    else:
        df["contribution_pct"] = df["contribution"] / total_abs * 100.0
        df["abs_pct"] = df["abs_contrib"] / total_abs * 100.0
    top3 = df.sort_values("abs_contrib", ascending=False).head(3)
    recs = []
    for r in top3.to_dict(orient="records"):
        recs.append({
            "feature": r["feature"],
            "pretty_name": pretty_feature_name(r["feature"]),
            "coefficient": float(r["coefficient"]),
            "value": float(r["value"]),
            "contribution": float(r["contribution"]),
            "abs_contribution": float(abs(r["contribution"])),
            "contribution_pct": float(r["contribution_pct"]),
            "abs_pct": float(r["abs_pct"])
        })
    return recs

# Replace the previous plotting functions with these improved versions.
def _short_label(s: str, width: int = 20):
    s = pretty_feature_name(s)
    if len(s) <= width:
        return s
    return "\n".join(textwrap.wrap(s, width=width))

def plot_top3_and_b64_abs(top3):
    """Plot absolute contributions with safe top padding and robust annotations."""
    feat = [_short_label(t["pretty_name"], width=18) for t in top3]
    vals = [t["abs_contribution"] for t in top3]
    colors = ['tomato' for _ in vals]

    # avoid zero-scale plots
    maxv = max(vals) if vals else 1.0
    top_pad_factor = 1.12 if maxv > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 4), dpi=160)
    bars = ax.bar(feat, vals, color=colors)
    ax.set_title("Top 3 Feature Contribution Magnitudes (absolute)", fontsize=12)
    ax.set_ylabel("Absolute contribution", fontsize=10)
    ax.set_xlabel("Feature", fontsize=10)

    # give headroom so annotations are not clipped
    y_max = maxv * top_pad_factor
    if y_max == 0:
        y_max = 1.0
    ax.set_ylim(0, y_max)

    # annotate bars: place inside if tall enough, else above (clip_on=False prevents clipping)
    for bar, t in zip(bars, top3):
        y = bar.get_height()
        pct_label = f"{t['abs_pct']:.1f}%"
        # determine threshold (12% of max) to place label inside bar
        inside_thresh = 0.12 * maxv if maxv>0 else 0.1
        if y >= inside_thresh:
            # inside center (contrast: white text)
            ax.text(bar.get_x() + bar.get_width()/2, y*0.5, pct_label, ha='center', va='center', fontsize=9, color='white', weight='bold', clip_on=False)
        else:
            # above bar (ensure it's visible by setting clip_on=False)
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.03 * maxv if maxv>0 else 0.03), pct_label, ha='center', va='bottom', fontsize=9, clip_on=False)

    plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22, top=0.92)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_group_and_b64_abs(group_list, all_features_df):
    """
    Plot group features with larger figure and safe annotations/limits.
    Returns (base64_png, rows_list).
    """
    # find matches like before (group_list are sanitized keys)
    san_features = [sanitize_colname(x) for x in all_features_df["feature"].tolist()]

    matches = []
    for want in group_list:
        for orig in all_features_df["feature"].tolist():
            if want in sanitize_colname(orig):
                if orig not in matches:
                    matches.append(orig)
        if not any(want in sanitize_colname(m) for m in matches):
            close = difflib.get_close_matches(want, san_features, n=3, cutoff=0.6)
            for cl in close:
                idx = san_features.index(cl)
                orig = all_features_df["feature"].tolist()[idx]
                if orig not in matches:
                    matches.append(orig)

    if not matches:
        return None, []

    rows = all_features_df[all_features_df["feature"].isin(matches)].copy()
    if rows.empty:
        return None, []

    rows = rows.sort_values("abs_contrib", ascending=False)

    pretty_names = [pretty_feature_name(f) for f in rows["feature"].tolist()]
    wrapped = [_short_label(n, width=18) for n in pretty_names]
    vals = rows["abs_contrib"].tolist()
    abs_pcts = rows["abs_pct"].tolist()

    maxv = max(vals) if vals else 1.0
    top_pad_factor = 1.12 if maxv > 0 else 1.0

    # dynamic width
    width = max(6, 0.9 * len(wrapped))
    fig, ax = plt.subplots(figsize=(width, 3.6), dpi=160)
    bars = ax.bar(wrapped, vals, color=['tomato' for _ in vals])

    ax.set_title("Contribution magnitudes (absolute)", fontsize=12)
    ax.set_ylabel("Absolute contribution", fontsize=10)
    ax.set_xlabel("Feature", fontsize=10)

    # set ylim to provide headroom
    y_max = maxv * top_pad_factor
    if y_max == 0:
        y_max = 1.0
    ax.set_ylim(0, y_max)

    # rotate slightly when necessary
    if len(wrapped) > 4 or max(len(x) for x in wrapped) > 12:
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right', fontsize=9)
    else:
        plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)

    # annotate with either inside label or above label (clip_on=False)
    for bar, pct in zip(bars, abs_pcts):
        y = bar.get_height()
        inside_thresh = 0.12 * maxv if maxv>0 else 0.1
        if y >= inside_thresh:
            ax.text(bar.get_x() + bar.get_width()/2, y*0.5, f"{pct:.1f}%", ha='center', va='center', fontsize=9, color='white', weight='bold', clip_on=False)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, y + (0.03 * maxv if maxv>0 else 0.03), f"{pct:.1f}%", ha='center', va='bottom', fontsize=9, clip_on=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28, top=0.92)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=160, bbox_inches='tight', pad_inches=0.30)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8'), rows.to_dict(orient="records")

def compute_top3_and_full_df(coef: np.ndarray, feature_list: List[str], x_vector: np.ndarray):
    """
    Compute per-feature contribution = coef * x_value and return:
      - top3: list of dictionaries for the top 3 features by abs contribution
      - full_df: pandas.DataFrame containing feature, coefficient, value, contribution, abs_contrib, abs_pct, contribution_pct

    Defensive behavior:
      - If lengths mismatch between coef, feature_list and x_vector, we truncate to the minimum common length.
      - All numeric outputs are cast to Python floats (serializable).
    """
    # ensure numpy arrays
    arr_coef = np.asarray(coef).reshape(-1)
    x_vec = np.asarray(x_vector).reshape(-1)

    # ensure feature_list is a list of strings
    feature_list = list(feature_list)

    # align lengths: use the minimum length among the three
    L = min(len(arr_coef), len(x_vec), len(feature_list))
    if L == 0:
        # nothing to compute
        empty_df = pd.DataFrame(columns=["feature", "coefficient", "value", "contribution", "abs_contrib", "abs_pct", "contribution_pct"])
        return [], empty_df

    arr_coef = arr_coef[:L]
    x_vec = x_vec[:L]
    feature_list = feature_list[:L]

    contrib = arr_coef * x_vec
    df = pd.DataFrame({
        "feature": feature_list,
        "coefficient": arr_coef.astype(float),
        "value": x_vec.astype(float),
        "contribution": contrib.astype(float),
    })
    df["abs_contrib"] = df["contribution"].abs()

    total_abs = float(df["abs_contrib"].sum())
    if total_abs > 0:
        df["abs_pct"] = (df["abs_contrib"] / total_abs * 100.0)
        df["contribution_pct"] = (df["contribution"] / total_abs * 100.0)
    else:
        df["abs_pct"] = 0.0
        df["contribution_pct"] = 0.0

    # sort and pick top3 by absolute contribution
    top3_df = df.sort_values("abs_contrib", ascending=False).head(3)

    top3 = []
    for r in top3_df.to_dict(orient="records"):
        pretty = r.get("feature")
        try:
            # use pretty_feature_name if available in your file
            pretty = pretty_feature_name(r.get("feature"))
        except Exception:
            pass
        top3.append({
            "feature": r.get("feature"),
            "pretty_name": pretty,
            "coefficient": float(r.get("coefficient", 0.0)),
            "value": float(r.get("value", 0.0)),
            "contribution": float(r.get("contribution", 0.0)),
            "abs_contribution": float(r.get("abs_contrib", 0.0)),
            "abs_pct": float(r.get("abs_pct", 0.0)),
            "contribution_pct": float(r.get("contribution_pct", 0.0))
        })

    return top3, df

# Robust LLM call helper (tries multiple interfaces)
async def call_offline_llm(prompt: str, max_tokens: int = 200):
    if llm is None:
        raise RuntimeError("Offline LLM not available")
    def do_call():
        try:
            if hasattr(llm, "create"):
                return llm.create(prompt=prompt, max_tokens=max_tokens)
        except Exception:
            pass
        try:
            if callable(llm):
                return llm(prompt=prompt, max_tokens=max_tokens)
        except Exception:
            pass
        try:
            if hasattr(llm, "generate"):
                return llm.generate(prompt=prompt, max_tokens=max_tokens)
        except Exception:
            pass
        raise RuntimeError("LLM has no supported interface")
    loop = asyncio.get_event_loop()
    raw_out = await loop.run_in_executor(executor, do_call)
    # normalize output to text
    text = None
    if isinstance(raw_out, dict):
        text = raw_out.get("response_text") or raw_out.get("response") or (raw_out.get("choices") and raw_out["choices"][0].get("text")) or raw_out.get("text")
    elif isinstance(raw_out, str):
        text = raw_out
    else:
        if hasattr(raw_out, "text"):
            text = getattr(raw_out, "text")
        elif hasattr(raw_out, "choices") and len(getattr(raw_out, "choices"))>0:
            c = getattr(raw_out, "choices")[0]
            text = getattr(c, "text", str(c))
        else:
            text = str(raw_out)
    return text.strip() if text else None

# Helper: build a final_report string (LLM if possible, deterministic fallback otherwise)
async def build_final_report(top3: List[Dict[str, Any]], disease_reports: List[Dict[str, Any]]):
    """
    Build a one-sentence Summary + 3 short Actionable steps.
    Prefer offline LLM if available; otherwise deterministic fallback.
    Always returns a non-empty string.
    """
    # Prepare short context lines
    top3_lines = []
    for t in top3:
        name = t.get("pretty_name") or t.get("feature")
        top3_lines.append(f"{name} = {t.get('value', 0):.2f} ({t.get('abs_pct', 0):.1f}% signal)")

    disease_lines = []
    for dr in disease_reports:
        # include disease name and short summary if available
        snippet = dr.get("llm_summary") or ""
        # keep the first 140 chars of the disease summary to avoid giant prompts
        snippet_short = (snippet.strip().splitlines()[0])[:140] if snippet else ""
        disease_lines.append(f"{dr['disease']}: {snippet_short}")

    # Build strict prompt that asks for two labeled sections
    prompt = (
        "You are a concise clinical assistant. Output only plain text. "
        "Produce exactly two sections labeled 'Summary:' and 'Actionable steps:'. "
        "Summary should be ONE sentence that names the main driver and whether it increases or decreases risk. "
        "Actionable steps should be three short recommendations, each <=10 words, each on a new line prefixed by '- '.\n\n"
        "Context - Top features:\n - " + ("\n - ".join(top3_lines) if top3_lines else "(none)") + "\n\n"
        "Context - Disease snippets:\n - " + ("\n - ".join(disease_lines) if disease_lines else "(none)") + "\n\n"
        "Now output the two sections only."
    )

    # Try the offline LLM (if available)
    llm_text = None
    if llm is not None:
        try:
            llm_text = await call_offline_llm(prompt, max_tokens=220)
            if llm_text:
                # sanitize — remove repeated paragraphs and trailing whitespace
                llm_text = llm_text.strip()
                # VERY common LLM verbosity: if it includes HTML or instructions, discard and fallback
                if len(llm_text) < 8 or any(tok.lower() in llm_text.lower() for tok in ["you may use", "<h1>", "step", "1/"]):
                    print("LLM output appears noisy; using fallback instead.")
                    llm_text = None
                else:
                    print("LLM produced final_report (len=%d)." % len(llm_text))
        except Exception as e:
            print("LLM call failed while building final_report:", e)
            llm_text = None

    if llm_text:
        return llm_text

    # deterministic fallback: pick primary driver and heuristics for actions
    if top3 and len(top3) > 0:
        primary = top3[0]
        pname = primary.get("pretty_name") or primary.get("feature")
        dir_word = "increasing" if primary.get("contribution", 0) > 0 else "decreasing"
        summary = f"Summary: {pname} is {dir_word} risk (primary driver)."
    else:
        summary = "Summary: No dominant risk driver identified."

    # assemble 3 short actions from disease reports or top features
    recs = []
    # try disease based heuristics first
    for dr in disease_reports:
        for f in dr.get("features", []):
            fname = f.get("feature", "")
            if "glucose" in fname and "Reduce simple sugars." not in recs:
                recs.append("Reduce simple sugars.")
            if "cholesterol" in fname and "Limit saturated fats." not in recs:
                recs.append("Limit saturated fats.")
            if ("bp" in fname or "blood_pressure" in fname) and "Measure BP regularly." not in recs:
                recs.append("Measure BP regularly.")
            if "pollution" in fname and "Avoid heavy pollution exposure." not in recs:
                recs.append("Avoid heavy pollution exposure.")
            if "smoker" in fname and "Stop smoking." not in recs:
                recs.append("Stop smoking.")

            if len(recs) >= 3:
                break
        if len(recs) >= 3:
            break

    # fallback fill
    while len(recs) < 3:
        recs.append("Review measurements and repeat testing.")

    actionable = "Actionable steps:\n- " + "\n- ".join(recs[:3])

    final = summary + "\n\n" + actionable
    return final


# disease groups mapping: keys are canonical feature names (sanitized)
DISEASE_GROUPS_RAW = {
    "Heart disease": ["bp_avg_7d","bp_days_above_140_last7","bp_volatility_7d","cholesterol_avg_7d","cholesterol_volatility_7d","cholesterol_days_above_200_last7","chest_pain_type"],
    "Diabetes": ["glucose_avg_7d","glucose_volatility_7d","glucose_days_above_180_last7","hr_days_above_100_last7"],
    "Asthma": ["family_history_asthma","smoker","wheezing_problem","pollution_index_latest"],
    "General": ["age","sex_m","bmi","sleep_avg_7d","sleep_volatility_7d","sleep_days_below_6h_last7"]
}
# sanitize disease keys mapping to actual feature names found
DISEASE_GROUPS = {k: [sanitize_colname(x) for x in v] for k,v in DISEASE_GROUPS_RAW.items()}

# ---------- PREDICT endpoint ----------
@app.post("/predict")
async def predict(csv_file: UploadFile = File(...)):
    """
    Predict endpoint — guaranteed to return a 'final_report' string per row.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server. Set MODEL_PATH and restart.")

    content = await csv_file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    if 'patient_id' in df.columns:
        grouping = df.groupby('patient_id')
    else:
        grouping = [(None, df)]

    output_rows = []
    for pid, group in grouping:
        derived = compute_derived_from_timeseries(group)
        feature_list = FEATURE_NAMES if FEATURE_NAMES else list(derived.keys())
        x_vec = np.array([derived.get(f, 0.0) for f in feature_list], dtype=float).reshape(1, -1)

        Xpd = pd.DataFrame(x_vec, columns=feature_list)
        X_aligned = align_X_to_model(Xpd)
        X_aligned = X_aligned.apply(pd.to_numeric, errors='coerce').fillna(0.0)

        try:
            preds = model.predict(X_aligned)
            probs = model.predict_proba(X_aligned) if hasattr(model, "predict_proba") else None
        except Exception as e:
            try:
                preds = model.predict(x_vec)
                probs = model.predict_proba(x_vec) if hasattr(model, "predict_proba") else None
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}; fallback failed: {e2}")

        if not hasattr(model, "coef_"):
            raise HTTPException(status_code=500, detail="Model has no coef_ for contribution computation.")
        coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_

        top3, full_df = compute_top3_and_full_df(coef, list(X_aligned.columns), X_aligned.values.flatten())
        chart_b64 = plot_top3_and_b64_abs(top3)

        # build disease reports as before (keeps the same behavior)
        disease_reports = []
        for disease_name, want_list in DISEASE_GROUPS.items():
            g_chart, rows = plot_group_and_b64_abs(want_list, full_df)
            group_features = []
            if rows:
                for rec in rows:
                    group_features.append({
                        "feature": rec["feature"],
                        "pretty_name": pretty_feature_name(rec["feature"]),
                        "value": float(rec["value"]),
                        "contribution": float(rec["contribution"]),
                        "abs_contribution": float(rec["abs_contrib"]),
                        "abs_pct": float(rec["abs_pct"])
                    })
            # LLM per-disease summary (attempt; fallback included in earlier code)
            llm_summary = None
            if group_features:
                disease_lines = [f"{g['pretty_name']} = {g['value']:.3f} ({g['abs_pct']:.1f}% of signal)" for g in group_features]
                general_matches = []
                for gk in DISEASE_GROUPS["General"]:
                    san_list = [sanitize_colname(f) for f in full_df["feature"].tolist()]
                    if gk in san_list:
                        idx = san_list.index(gk)
                        row = full_df.iloc[idx]
                        general_matches.append(f"{pretty_feature_name(row['feature'])} = {float(row['value']):.3f} ({float(row['abs_pct']):.1f}% of signal)")
                prompt = (
                    "You are a concise clinical assistant. Output only plain text. Produce exactly two labeled sections: Interpretation and Prevention.\n\n"
                    f"Disease: {disease_name}\n"
                    "General context:\n - " + ("\n - ".join(general_matches) if general_matches else "(none)") + "\n"
                    "Disease features:\n - " + "\n - ".join(disease_lines) + "\n\n"
                    "Interpretation: one sentence naming the main driver and whether it increases or decreases risk. Prevention: two short actionable recommendations (<=10 words each). Do not mention timeframes.\n"
                )
                if AUTO_RUN_LLM and (llm is not None):
                    try:
                        llm_text = await call_offline_llm(prompt, max_tokens=120)
                        if llm_text:
                            # sanitize minor noise
                            llm_summary = llm_text.strip()
                            # if the output seems obviously system/instructional, drop it to fallback logic below
                            if len(llm_summary) < 8 or any(kw in llm_summary.lower() for kw in ["you may use", "<h1>", "step", "1/"]):
                                print(f"Per-disease LLM returned noisy text for {disease_name}; dropping to fallback.")
                                llm_summary = None
                    except Exception as e:
                        print("Per-disease LLM call failed:", e)
                        llm_summary = None

            # fallback if no llm_summary
            if not llm_summary:
                if group_features:
                    primary = sorted(group_features, key=lambda x: x["abs_contribution"], reverse=True)[0]
                    dir_word = "increases" if primary["contribution"] > 0 else "decreases"
                    interp = f"{primary['pretty_name']} {dir_word} risk (primary driver)."
                    # simple heuristics for prevention
                    if "glucose" in primary["feature"]:
                        recs = "- Reduce simple sugars.\n- Monitor glucose regularly."
                    elif "cholesterol" in primary["feature"]:
                        recs = "- Limit saturated fats.\n- Repeat lipid panel."
                    elif "bp" in primary["feature"]:
                        recs = "- Measure blood pressure twice daily.\n- Review antihypertensive therapy."
                    else:
                        recs = "- Review measurement.\n- Repeat and reassess."
                    llm_summary = "Interpretation: " + interp + "\n\nPrevention:\n" + recs
                else:
                    llm_summary = f"No disease-specific features present for {disease_name}."

            disease_reports.append({
                "disease": disease_name,
                "features": group_features,
                "chart_png_base64": g_chart,
                "llm_summary": llm_summary
            })

        # Build final report (guaranteed)
        final_report = await build_final_report(top3, disease_reports)

        output_rows.append({
            "patient_id": pid,
            "prediction": int(preds[0]),
            "probabilities": (probs[0].tolist() if probs is not None else None),
            "confidence": float(probs[0][int(preds[0])]) if (probs is not None) else None,
            "derived_input": {f: float(derived.get(f, 0.0)) for f in feature_list},
            "feature_list": feature_list,
            "top3_contributions": top3,
            "chart_png_base64": chart_b64,
            "disease_reports": disease_reports,
            "final_report": final_report
        })

    return {"rows": output_rows, "feature_names_used": FEATURE_NAMES}

@app.post("/llm")
async def call_llm(payload: dict):
    if llm is None:
        raise HTTPException(status_code=503, detail="Offline LLM not available on server.")
    prompt = payload.get("prompt") or payload.get("text")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in payload.")
    max_tokens = int(payload.get("max_tokens", LLM_MAX_TOKENS))
    try:
        txt = await call_offline_llm(prompt, max_tokens=max_tokens)
        return {"response_text": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM inference error: {e}")


if __name__ == "__main__":
    print("Model present:", model is not None)
    if model is not None and hasattr(model, "feature_names_in_"):
        print("Model.feature_names_in_ (sample 20):", list(model.feature_names_in_)[:20])
    else:
        print("Model.feature_names_in_ not available")
    print("LLM loaded:", llm is not None)
    print("AUTO_RUN_LLM:", AUTO_RUN_LLM)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)