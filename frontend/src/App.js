import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function handleUpload(e) {
    e.preventDefault();
    if (!file) return alert("Please choose a CSV file");
    setLoading(true);
    setResult(null);
    const fd = new FormData();
    fd.append("csv_file", file);
    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: fd,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
      const rows = Array.isArray(data.rows) ? data.rows : [];
      setResult({ data, row: rows.length ? rows[0] : null });
    } catch (err) {
      console.error(err);
      alert("Upload error: " + (err.message || err));
    } finally {
      setLoading(false);
    }
  }

  const row = result?.row ?? null;

  function formatConfidence(r) {
    if (!r || typeof r.confidence === "undefined" || r.confidence === null) return "N/A";
    return Number(r.confidence).toFixed(3);
  }

  function downloadReport() {
    const text = row?.final_report ?? row?.llm_response ?? "No report available.";
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${row?.patient_id ?? "report"}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="app-root">
      <header className="topbar">
        <div className="brand">
          <div className="brand-logo">🫀</div>
          <div>
            <h1>Patient Predictor</h1>
            <div className="subtitle">Risk prediction & disease-specific summaries</div>
          </div>
        </div>
        <div className="uploader">
          <label className="file-btn">
            <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files[0])} />
            <span>{file ? file.name : "Choose CSV"}</span>
          </label>
          <button className="primary" onClick={handleUpload} disabled={loading}>
            {loading ? "Running..." : "Upload & Predict"}
          </button>
        </div>
      </header>

      {!result && (
        <main className="main">
          <div className="placeholder-card">
            <h3>No results yet</h3>
            <p>Upload a patient's 90-day CSV (time-series) to generate prediction, top features and disease reports.</p>
          </div>
        </main>
      )}

      {result && row && (
        <main className="main grid-2">
          <section className="card summary-card">
            <div className="summary-header">
              <div>
                <div className="pill">Patient</div>
                <h2>{row.patient_id ?? "N/A"}</h2>
              </div>
              <div className="summary-actions">
                <button className="ghost" onClick={downloadReport}>Download Report</button>
              </div>
            </div>

            <div className="summary-meta">
              <div><span className="meta-title">Prediction</span> <span className="meta-value">{row.prediction === 1 ? "Deteriorated (1)" : "Healthy (0)"}</span></div>
              <div><span className="meta-title">Confidence</span> <span className="meta-value">{formatConfidence(row)}</span></div>
            </div>

            <div className="summary-box">
              <h3>Summary</h3>
              <div className="summary-text">
                {row.final_report ?? row.llm_response ?? "No report available."}
              </div>
            </div>

            <div className="top3-row">
              <h4>Top contributors</h4>
              <div className="top3-list">
                {(row.top3_contributions ?? []).map((t, idx) => (
                  <div key={idx} className="top3-item">
                    <div className="top3-name">{t.pretty_name ?? t.feature}</div>
                    <div className="top3-meta">
                      <span className="small muted">abs: {Number(t.abs_contribution).toFixed(3)}</span>
                      <span className="small muted">{Number(t.abs_pct).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <aside className="card chart-card">
            <h3>Top 3 Contribution Chart</h3>
            <div className="chart-wrap">
              {row.chart_png_base64 ? (
                <img alt="top3 chart" src={`data:image/png;base64,${row.chart_png_base64}`} />
              ) : (
                <div className="muted">Chart not available</div>
              )}
            </div>

            <div className="legend">
              <div className="legend-item"><span className="dot red" /> Positive contribution</div>
              <div className="legend-item"><span className="dot blue" /> Negative contribution (magnitude shown separately)</div>
            </div>
          </aside>

          <section className="card full-width">
            <h3>Disease Reports</h3>
            <div className="disease-grid">
              {(row.disease_reports || []).map((dr, idx) => (
              <div key={idx} className="disease-card">
                {/* chart at the top */}
                {dr.chart_png_base64 ? (
                  <div className="disease-chart-top">
                    <img className="mini-chart-large" alt={`${dr.disease} chart`} src={`data:image/png;base64,${dr.chart_png_base64}`} />
                  </div>
                ) : null}

                <div className="disease-head">
                  <h4>{dr.disease}</h4>
                </div>

                <div className="disease-features">
                  <div className="muted small">Features (abs % of total)</div>
                  <ul>
                    {dr.features && dr.features.length > 0 ? dr.features.map((f, i2) => (
                      <li key={i2}>
                        <strong>{f.pretty_name}</strong> <span className="small muted">— {Number(f.abs_pct).toFixed(1)}%</span>
                      </li>
                    )) : <li className="muted">No features present</li>}
                  </ul>
                </div>

                <div className="disease-summary">
                  <div className="muted small">Interpretation & Prevention</div>
                  <div className="disease-text">{dr.llm_summary ?? "No summary available."}</div>
                </div>
              </div>
            ))}
            </div>
          </section>
        </main>
      )}

      <footer className="footer">
        <div>Made with ♥ — Patient Predictor</div>
        <div className="footer-right">Model: local | UI: stylized</div>
      </footer>
    </div>
  );
}

export default App;
