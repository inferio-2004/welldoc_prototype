# backend/app.py
import os
import io
import json
import base64
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import aiohttp
import asyncio
import json
import concurrent.futures
app = FastAPI(title="Patient Predictor API (Derived-features + 7d aggregation)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "logistic_model.pkl")
FEATURES_TXT = os.environ.get("FEATURES_TXT", "derived_cols.txt")

# load model
try:
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Warning: could not load model at {MODEL_PATH}: {e}")


# ---------- Utility helpers ----------
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


# ---------- Load canonical derived features list (sanitized) ----------
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
    # fallback: try model feature names
    if model is not None and hasattr(model, "feature_names_in_"):
        FEATURE_NAMES = [sanitize_colname(x) for x in list(model.feature_names_in_)]
        print("Using model.feature_names_in_ (sanitized).")
    else:
        print("No derived features file and model.feature_names_in_ not found. Will infer from CSV.")


# ---------- Derived feature builders ----------
# Exact derived names expected (sanitized) are in FEATURE_NAMES; we map rules below.
# We'll compute for each patient (group by patient_id if present) a single derived row.

# Helper: compute slope (linear fit) over last_n days
def compute_slope(series: pd.Series) -> float:
    # returns slope per day (last point - first point)/n if not enough points, else np.polyfit
    y = series.dropna().astype(float).values
    n = len(y)
    if n <= 1:
        return 0.0
    x = np.arange(n)
    # linear fit
    try:
        m, b = np.polyfit(x, y, 1)
        return float(m)
    except Exception:
        # fallback simple difference
        return float((y[-1] - y[0]) / (n - 1))


def compute_derived_from_timeseries(df_patient: pd.DataFrame) -> Dict[str, float]:
    """
    df_patient: rows for a single patient (time-ordered or unordered). Expect day/date column optional.
    Returns dict with keys equal to FEATURE_NAMES (sanitized).
    """
    out = {}
    # ensure sorted by day if 'day' column exists; else keep as-is but use last rows
    if 'day' in df_patient.columns:
        df_patient = df_patient.sort_values('day')
    # For rolling 7-day calculations, take last 7 rows (or fewer if not available)
    last7 = df_patient.tail(7)

    # Utility to read candidate raw columns by various possible names
    def find_col_like(df, patterns: List[str]):
        for c in df.columns:
            lc = c.lower()
            for p in patterns:
                if p in lc:
                    return c
        return None

    # Map raw column names
    col_chol = find_col_like(df_patient, ["cholesterol", "chol"])
    col_glucose = find_col_like(df_patient, ["glucose", "sugar"])
    col_bp_sys = find_col_like(df_patient, ["systolic", "systolic_bp", "systolic bp", "sbp"])
    col_bp_dia = find_col_like(df_patient, ["diastolic", "diastolic_bp", "dbp"])
    col_bp = col_bp_sys if col_bp_sys is not None else find_col_like(df_patient, ["blood pressure", "bp"])
    col_weight = find_col_like(df_patient, ["weight", "wt"])
    col_spo2 = find_col_like(df_patient, ["spo2", "o2", "oxygen", "saturation"])
    col_hr = find_col_like(df_patient, ["heart_rate", "heart rate", "hr", "pulse"])
    col_sleep = find_col_like(df_patient, ["sleep", "sleep_hours", "sleephours"])
    col_bmi = find_col_like(df_patient, ["bmi"])
    col_pollution = find_col_like(df_patient, ["pollution", "pi", "pollution_index"])
    col_smoker = find_col_like(df_patient, ["smoker", "smoking"])
    col_asthma = find_col_like(df_patient, ["asthma"])
    col_diabetes = find_col_like(df_patient, ["diabetes"])
    col_hd = find_col_like(df_patient, ["heart disease", "heart_disease", "heartdisease", "heart"])
    col_family = find_col_like(df_patient, ["family", "family_history", "family history"])
    col_wheezing = find_col_like(df_patient, ["wheezing", "wheeze"])
    col_sex = find_col_like(df_patient, ["sex", "gender"])
    col_chest = find_col_like(df_patient, ["chest", "chest pain", "chest_pain", "chestpain"])

    # read last/latest scalars
    def safe_last(col):
        if col is None or col not in df_patient.columns:
            return np.nan
        return to_float_safe(df_patient[col].dropna().astype(float).values[-1]) if df_patient[col].dropna().size>0 else np.nan

    # cholesterol
    chol_vals = pd.to_numeric(last7[col_chol], errors='coerce') if col_chol else pd.Series(dtype=float)
    chol_latest = safe_last(col_chol)
    out["cholesterol_latest"] = float(np.nan_to_num(chol_latest, 0.0))
    out["cholesterol_avg_7d"] = float(np.nan_to_num(chol_vals.mean(), nan=0.0))
    out["cholesterol_slope_7d"] = float(compute_slope(chol_vals)) if len(chol_vals)>0 else 0.0
    out["cholesterol_volatility_7d"] = float(np.nan_to_num(chol_vals.std(ddof=0), 0.0))
    out["cholesterol_days_above_200_last7"] = float(np.sum(chol_vals > 200)) if len(chol_vals)>0 else 0.0

    # glucose
    gl_vals = pd.to_numeric(last7[col_glucose], errors='coerce') if col_glucose else pd.Series(dtype=float)
    gl_latest = safe_last(col_glucose)
    out["glucose_latest"] = float(np.nan_to_num(gl_latest, 0.0))
    out["glucose_avg_7d"] = float(np.nan_to_num(gl_vals.mean(), 0.0))
    out["glucose_slope_7d"] = float(compute_slope(gl_vals)) if len(gl_vals)>0 else 0.0
    out["glucose_volatility_7d"] = float(np.nan_to_num(gl_vals.std(ddof=0), 0.0))
    out["glucose_days_above_180_last7"] = float(np.sum(gl_vals > 180)) if len(gl_vals)>0 else 0.0

    # bp (use systolic if available)
    bp_vals = pd.to_numeric(last7[col_bp], errors='coerce') if col_bp else pd.Series(dtype=float)
    bp_latest = safe_last(col_bp)
    out["bp_latest"] = float(np.nan_to_num(bp_latest, 0.0))
    out["bp_avg_7d"] = float(np.nan_to_num(bp_vals.mean(), 0.0))
    out["bp_slope_7d"] = float(compute_slope(bp_vals)) if len(bp_vals)>0 else 0.0
    out["bp_volatility_7d"] = float(np.nan_to_num(bp_vals.std(ddof=0), 0.0))
    out["bp_days_above_140_last7"] = float(np.sum(bp_vals > 140)) if len(bp_vals)>0 else 0.0

    # weight
    wt_vals = pd.to_numeric(last7[col_weight], errors='coerce') if col_weight else pd.Series(dtype=float)
    wt_latest = safe_last(col_weight)
    out["weight_latest"] = float(np.nan_to_num(wt_latest, 0.0))
    out["weight_avg_7d"] = float(np.nan_to_num(wt_vals.mean(), 0.0))
    out["weight_volatility_7d"] = float(np.nan_to_num(wt_vals.std(ddof=0), 0.0))

    # spo2
    spo2_vals = pd.to_numeric(last7[col_spo2], errors='coerce') if col_spo2 else pd.Series(dtype=float)
    spo2_latest = safe_last(col_spo2)
    out["spo2_latest"] = float(np.nan_to_num(spo2_latest, 0.0))
    out["spo2_avg_7d"] = float(np.nan_to_num(spo2_vals.mean(), 0.0))
    out["spo2_min_7d"] = float(np.nan_to_num(spo2_vals.min(), 0.0))
    out["spo2_slope_7d"] = float(compute_slope(spo2_vals)) if len(spo2_vals)>0 else 0.0
    out["spo2_days_below_92_last7"] = float(np.sum(spo2_vals < 92)) if len(spo2_vals)>0 else 0.0

    # heart rate
    hr_vals = pd.to_numeric(last7[col_hr], errors='coerce') if col_hr else pd.Series(dtype=float)
    hr_latest = safe_last(col_hr)
    out["hr_latest"] = float(np.nan_to_num(hr_latest, 0.0))
    out["hr_avg_7d"] = float(np.nan_to_num(hr_vals.mean(), 0.0))
    out["hr_slope_7d"] = float(compute_slope(hr_vals)) if len(hr_vals)>0 else 0.0
    out["hr_volatility_7d"] = float(np.nan_to_num(hr_vals.std(ddof=0), 0.0))
    out["hr_days_above_100_last7"] = float(np.sum(hr_vals > 100)) if len(hr_vals)>0 else 0.0

    # sleep
    sleep_vals = pd.to_numeric(last7[col_sleep], errors='coerce') if col_sleep else pd.Series(dtype=float)
    sleep_latest = safe_last(col_sleep)
    out["sleep_latest"] = float(np.nan_to_num(sleep_latest, 0.0))
    out["sleep_avg_7d"] = float(np.nan_to_num(sleep_vals.mean(), 0.0))
    out["sleep_slope_7d"] = float(compute_slope(sleep_vals)) if len(sleep_vals)>0 else 0.0
    out["sleep_volatility_7d"] = float(np.nan_to_num(sleep_vals.std(ddof=0), 0.0))
    out["sleep_days_below_6h_last7"] = float(np.sum(sleep_vals < 6)) if len(sleep_vals)>0 else 0.0

    # pollution
    poll_vals = pd.to_numeric(last7[col_pollution], errors='coerce') if col_pollution else pd.Series(dtype=float)
    poll_latest = safe_last(col_pollution)
    out["pollution_index_latest"] = float(np.nan_to_num(poll_latest, 0.0))

    # wheezing
    wheeze_flag = find_col_like(df_patient, ["wheezing", "wheeze"])
    out["wheezing_problem"] = float(parse_yes_no(df_patient[wheeze_flag].dropna().iloc[-1])) if (wheeze_flag and df_patient[wheeze_flag].dropna().size>0) else 0.0

    # bmi (attempt direct or compute from weight/height if provided - height not implemented)
    out["bmi"] = float(np.nan_to_num(safe_last(col_bmi), 0.0))

    # binary flags and simple categorical
    out["smoker"] = float(parse_yes_no(df_patient[col_smoker].dropna().iloc[-1])) if (col_smoker and df_patient[col_smoker].dropna().size>0) else 0.0
    out["heart_disease"] = float(parse_yes_no(df_patient[col_hd].dropna().iloc[-1])) if (col_hd and df_patient[col_hd].dropna().size>0) else 0.0
    out["diabetes"] = float(parse_yes_no(df_patient[col_diabetes].dropna().iloc[-1])) if (col_diabetes and df_patient[col_diabetes].dropna().size>0) else 0.0
    out["asthma"] = float(parse_yes_no(df_patient[col_asthma].dropna().iloc[-1])) if (col_asthma and df_patient[col_asthma].dropna().size>0) else 0.0
    out["family_history_asthma"] = float(parse_yes_no(df_patient[col_family].dropna().iloc[-1])) if (col_family and df_patient[col_family].dropna().size>0) else 0.0

    # sex and chest pain (we keep original chest pain numeric as is)
    out["sex_m"] = 1.0 if (col_sex and str(df_patient[col_sex].dropna().iloc[-1]).strip().lower() in ("m","male","1")) else 0.0
    out["chest_pain_type"] = float(to_float_safe(df_patient[col_chest].dropna().iloc[-1])) if (col_chest and df_patient[col_chest].dropna().size>0) else 0.0

    # age (take latest)
    age_col = find_col_like(df_patient, ["age"])
    out["age"] = float(to_float_safe(df_patient[age_col].dropna().iloc[-1])) if (age_col and df_patient[age_col].dropna().size>0) else 0.0

    # ensure all FEATURE_NAMES present: if missing add 0.0
    if FEATURE_NAMES:
        for f in FEATURE_NAMES:
            if f not in out:
                out[f] = 0.0

    return out


# ---------- contribution, chart helpers ----------
def compute_top3_contributions_from_vector(coef: np.ndarray, feature_list: List[str], x_vector: np.ndarray):
    arr_coef = coef.reshape(-1)
    contrib = arr_coef * x_vector.reshape(-1)
    df = pd.DataFrame({
        "feature": feature_list,
        "coefficient": arr_coef,
        "value": x_vector.reshape(-1),
        "contribution": contrib
    })
    df["abs_contrib"] = df["contribution"].abs()
    top3 = df.sort_values("abs_contrib", ascending=False).head(3)
    return top3.to_dict(orient="records")


def plot_top3_and_b64(top3):
    feat = [t["feature"] for t in top3]
    vals = [t["contribution"] for t in top3]
    colors = ['tomato' if v > 0 else 'skyblue' for v in vals]
    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(feat, vals, color=colors)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title("Top 3 Feature Contributions (coef * value)")
    ax.set_ylabel("Contribution to linear score")
    ax.set_xlabel("Feature")
    for bar in bars:
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, y, round(y,4), ha='center', va='bottom' if y>=0 else 'top')
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ---------- main predict endpoint ----------
@app.post("/predict")
async def predict(csv_file: UploadFile = File(...)):
    """
    Accepts CSV that can contain either:
      - single-row per patient with raw fields, or
      - many rows per patient (time series). We group by 'patient_id' if present; otherwise treat whole CSV as one patient.
    Returns: derived row per patient, prediction, probabilities, top3 contributions and chart base64.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server. Set MODEL_PATH and restart.")

    content = await csv_file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # decide grouping key
    if 'patient_id' in df.columns:
        grouping = df.groupby('patient_id')
    else:
        # single anonymous group
        grouping = [(None, df)]

    output_rows = []
    # For later alignment to model expected order, we will use FEATURE_NAMES if available, else infer order
    for pid, group in grouping:
        # build derived dict
        derived = compute_derived_from_timeseries(group)
        # ensure order list
        feature_list = FEATURE_NAMES if FEATURE_NAMES else list(derived.keys())
        x_vec = np.array([derived.get(f, 0.0) for f in feature_list], dtype=float).reshape(1, -1)

        # call model.predict (ensure column ordering matches training: if model.feature_names_in_ exists, try to align)
        # NOTE: sklearn will warn/error if feature names mismatch — it's best if you trained with same names.
        try:
            Xpd = pd.DataFrame(x_vec, columns=feature_list)
            preds = model.predict(Xpd)
            probs = model.predict_proba(Xpd) if hasattr(model, "predict_proba") else None
        except Exception as e:
            # try fallback to raw numpy input (some models accept positional arrays)
            try:
                preds = model.predict(x_vec)
                probs = model.predict_proba(x_vec) if hasattr(model, "predict_proba") else None
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}; fallback failed: {e2}")

        # compute top3 contributions (needs coef_)
        if not hasattr(model, "coef_"):
            raise HTTPException(status_code=500, detail="Model has no coef_ for contribution computation.")
        coef = model.coef_[0] if model.coef_.ndim>1 else model.coef_
        top3 = compute_top3_contributions_from_vector(coef, feature_list, x_vec.flatten())
        chart_b64 = plot_top3_and_b64(top3)

        # prepare LLM-friendly summary (string)
        top3_text = "; ".join([f"{t['feature']} (contribution {t['contribution']:.4f})" for t in top3])
        llm_prompt = (
            f"Patient {pid if pid is not None else '(single)'} predicted as {int(preds[0])} with probs "
            f"{(probs[0].tolist() if probs is not None else None)}. Top contributing features: {top3_text}. "
            "Give a short explanation (2-3 bullets) on why this prediction likely occurred and recommended next checks."
        )

        output_rows.append({
            "patient_id": pid,
            "prediction": int(preds[0]),
            "probabilities": (probs[0].tolist() if probs is not None else None),
            "derived_input": {f: float(derived.get(f, 0.0)) for f in feature_list},
            "feature_list": feature_list,
            "top3_contributions": top3,
            "chart_png_base64": chart_b64,
            "llm_prompt": llm_prompt
        })

    return {"rows": output_rows, "feature_names_used": FEATURE_NAMES}


# ---------- simple LLM endpoint (example using llama-cpp-python) ----------
# If you prefer to call an external API, keep the older /llm proxy logic.
# The following assumes you installed llama-cpp-python and set LLM_GGUF_PATH env var.
try:
    from llama_cpp import Llama
    LLM_GGUF_PATH = os.environ.get("LLM_GGUF_PATH", "models\gemma-2b.Q4_K_M.gguf")
    llm = Llama(model_path=LLM_GGUF_PATH, temperature=0.2,verbose=False)
    print("Loaded offline GGUF LLM at", LLM_GGUF_PATH)
except Exception as e:
    llm = None
    print("Offline LLM not loaded:", e)

@app.post("/llm")
async def call_llm(payload: dict):
    """
    Call offline LLM. Payload expects 'prompt' (or 'text') and optional max_tokens.
    Returns: {"response_text": "..."}
    This function tries several common llama-cpp-python interfaces (create, __call__, generate).
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="Offline LLM not available on server.")
    prompt = payload.get("prompt") or payload.get("text")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in payload.")
    max_tokens = int(payload.get("max_tokens", LLM_MAX_TOKENS if 'LLM_MAX_TOKENS' in globals() else 250))

    def do_call():
        # Try .create(prompt=...) first (many examples)
        try:
            if hasattr(llm, "create"):
                return llm.create(prompt=prompt, max_tokens=max_tokens)
        except Exception:
            pass
        # Try callable object: llm(prompt=..., max_tokens=...)
        try:
            if callable(llm):
                return llm(prompt=prompt, max_tokens=max_tokens)
        except Exception:
            pass
        # Try .generate()
        try:
            if hasattr(llm, "generate"):
                return llm.generate(prompt=prompt, max_tokens=max_tokens)
        except Exception:
            pass
        raise RuntimeError("LLM object does not support create/prompt/generate interface on this installation.")

    try:
        loop = asyncio.get_event_loop()
        llm_out = await loop.run_in_executor(executor, do_call)

        # Normalize output to text
        text = ""
        if isinstance(llm_out, dict):
            # common keys
            text = llm_out.get("response_text") or llm_out.get("response") or \
                   (llm_out.get("choices") and llm_out["choices"][0].get("text")) or \
                   llm_out.get("text") or json.dumps(llm_out)
        elif isinstance(llm_out, str):
            text = llm_out
        else:
            # fallback: try to stringify object (some versions return LlamaResponse or Generation)
            try:
                # try common attribute names
                if hasattr(llm_out, "text"):
                    text = getattr(llm_out, "text")
                elif hasattr(llm_out, "choices") and len(llm_out.choices) > 0:
                    # choices may be objects
                    c = llm_out.choices[0]
                    text = getattr(c, "text", str(c))
                else:
                    text = str(llm_out)
            except Exception:
                text = str(llm_out)
        print("llm response: ",text)
        return {"response_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM inference error: {e}")


# Run
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
