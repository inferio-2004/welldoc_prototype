
# WellDoc Prototype 🩺

**Patient risk-prediction dashboard** — ingest 90-day vitals CSV, derive features, predict health deterioration with a trained Logistic Regression model, and explain results with disease-specific summaries powered by an offline LLM.

---

<img width="1903" height="943" alt="Screenshot 2025-09-06 215256" src="https://github.com/user-attachments/assets/ce8e2d31-60b2-418e-85cb-5f99e3eb761c" />
<img width="456" height="645" alt="image" src="https://github.com/user-attachments/assets/0f2ea3f9-2c33-4f2a-b72c-5bbde7da1304" />

Welldoc Prototype is a small, local web application that ingests a 90-day time-series CSV for a single patient, condenses the time series into derived features, runs a pre-trained logistic-regression model to predict deterioration risk (0 = healthy, 1 = deteriorated), computes per-feature contributions (and normalized contribution shares), and produces disease-specific charts and concise, clinician-friendly summaries using an offline LLM.

## This repo contains

1. **backend/** — FastAPI server that builds derived features, runs the model (.pkl), computes contributions, and (optionally) calls a local GGUF LLM for humanized summaries.  
2. **frontend/** — React UI for uploading CSVs, viewing predictions, top contributors, disease reports, and the final summary.  
3. **input_cols.txt / derived_cols.txt** — human-readable column definitions used to map raw inputs → derived features.  
4. **Instructions** to use Git LFS for large model binaries (.pkl, .gguf).  
5. This project is designed for prototyping and clinician demonstration — **not for production use** or standalone clinical decision making. Do not commit private patient data.  

---

## Key features

1. Accepts a CSV of daily vitals (up to 90 days) and groups by `patient_id` (or treats entire file as one patient).  
2. Generates derived features (latest, 7-day average, slopes, volatility, count of threshold breaches).  
3. Runs an already-trained Logistic Regression model (`model.pkl`) and returns prediction + confidence.  
4. Computes per-feature linear contributions (`coef * value`), ranks top-3, and shows normalized percentages.  
5. Produces per-disease charts (Heart disease, Diabetes, Asthma, General) and LLM-generated **Interpretation + Prevention** text.  
6. Optional offline LLM integration (GGUF via `llama-cpp-python` or equivalent) for natural language summaries.  
7. Stylized React UI with downloadable final report.  

---

## How it works (high level)

1. Frontend uploads a CSV to `/predict`.  
2. Backend reads CSV, groups rows, builds derived features per patient using heuristics and `derived_cols.txt`.  
3. Backend aligns features to the model, runs prediction and probability, computes per-feature contributions using `model.coef_`.  
4. Backend normalizes contributions to percentages, generates charts (PNG, base64), prepares disease-specific feature groups, and optionally calls the offline LLM to produce concise text summaries.  


# WellDoc Prototype 🩺

An end-to-end prototype health dashboard for patient risk prediction.  
The system ingests **90-day patient vitals**, computes **derived features**, predicts **deterioration risk** using a trained Logistic Regression model, and generates **human-readable explanations** via an offline LLM.

---

## 🚀 Features

- **Backend (FastAPI)**
  - Accepts raw patient CSV (90 days of vitals).
  - Derives features (e.g., averages, slopes, volatility).
  - Predicts deterioration risk using a trained Logistic Regression model (`model.pkl`).
  - Computes **top contributing features** for each prediction (feature attribution).
  - Integrates with an **offline LLM (GGUF)** to generate:
    - Report summary
    - Disease-specific interpretation (Heart Disease, Diabetes, Asthma, General factors).
  - Returns JSON responses + base64-encoded graphs.

- **Frontend (React + Vite / CRA)**
  - Upload CSV → get prediction dashboard.
  - Displays:
    - **Prediction result** (Healthy / Deteriorated)
    - **Confidence score**
    - **Top 3 feature contributions** (bar graph)
    - **Disease-specific analysis** with graphs + LLM-generated summaries
    - **Final report summary**
  - Clean, stylized UI.

---

## 🏗 Project Structure

```
welldoc_prototype/
│
├── backend/          # FastAPI app
│   ├── app.py        # Main API
│   ├── model.pkl     # Trained Logistic Regression model (LFS)
│   └── requirements.txt
│
├── frontend/         # React app (CRA)
│   ├── src/          # Components & UI
│   ├── public/
│   ├── package.json
│   └── ...
│
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate   # on Windows
pip install -r requirements.txt

# run API
uvicorn app:app --reload
```

Backend runs on: **http://localhost:8000**

---

### 2. Frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on: **http://localhost:3000**

---

### 3. Offline LLM (optional)

This project supports **offline inference** via GGUF models (e.g. MazharPayani’s Gemini models from HuggingFace).

Example setup:

```bash
cd backend
pip install llama-cpp-python

# download model
python download_gguf.py
```

Then configure `AUTO_RUN_LLM=True` in backend to auto-generate LLM summaries.

---

## 📊 Example Flow

1. Upload a CSV of patient vitals (90 days).
2. Backend condenses to **derived features**.
3. Logistic Regression predicts deterioration.
4. Dashboard shows:
   - Confidence score
   - Feature contribution graph
   - Disease-specific graphs + LLM interpretation
   - Final report summary

---

## 📝 Roadmap

- [ ] Add support for multiple patients per batch CSV
- [ ] Improve visualization (interactive charts)
- [ ] Fine-tune LLM prompts for more natural summaries
- [ ] Export report as PDF

---

## 📄 License

MIT License © 2025

---

## 👨‍💻 Authors

- **Aniruth [@inferio-2004](https://github.com/inferio-2004)** — Full-stack prototype, ML integration, ML training
- **Akil [@AKIL3333](https://github.com/AKIL3333)** — Full-stack prototype, ML integration, ML training
- **Thashventh [@thahsventh21](https://github.com/thahsventh21)** — Full-stack prototype, ML integration, ML training
- **Jagadesh [@JagadeeshTheJD](https://github.com/JagadeeshTheJD)** — Full-stack prototype, ML integration, ML training
