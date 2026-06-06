import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "https://madhuri1003-workpulse-api.hf.space";


function getSentimentColor(sentiment) {
  if (sentiment === "Positive") return "#22c55e";
  if (sentiment === "Negative") return "#ef4444";
  if (sentiment === "Mixed") return "#f59e0b";
  return "#6b7280";
}

function getSentimentEmoji(sentiment) {
  if (sentiment === "Positive") return "✅";
  if (sentiment === "Negative") return "🚨";
  if (sentiment === "Mixed") return "⚠️";
  return "➖";
}

function ScoreBar({ label, value }) {
  const color =
    label === "Positive" ? "#22c55e" :
    label === "Negative" ? "#ef4444" : "#6b7280";
  return (
    <div className="score-bar-row">
      <span className="score-label">{label}</span>
      <div className="score-bar-bg">
        <div
          className="score-bar-fill"
          style={{ width: `${(value * 100).toFixed(1)}%`, background: color }}
        />
      </div>
      <span className="score-pct">{(value * 100).toFixed(1)}%</span>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("single");
  const [batchText, setBatchText] = useState("");
  const [batchResults, setBatchResults] = useState(null);

  const analyzeSingle = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, { text });
      setResult(res.data);
    } catch (err) {
      setError("Backend not reachable. Make sure uvicorn is running.");
    }
    setLoading(false);
  };

  const analyzeBatch = async () => {
    const texts = batchText.split("\n").map(t => t.trim()).filter(Boolean);
    if (texts.length === 0) return;
    setLoading(true);
    setError(null);
    setBatchResults(null);
    try {
      const res = await axios.post(`${API_URL}/batch`, { texts });
      setBatchResults(res.data);
    } catch (err) {
      setError("Backend not reachable. Make sure uvicorn is running.");
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <header className="header">
        <h1>WorkPulse 🔍</h1>
        <p>Company Culture Sentiment Analyzer</p>
        <p className="subtext">Fine-tuned DistilBERT · 838K Glassdoor Reviews · 3-Class Classification</p>
      </header>

      <div className="mode-toggle">
        <button
          className={mode === "single" ? "toggle-btn active" : "toggle-btn"}
          onClick={() => setMode("single")}
        >Single Review</button>
        <button
          className={mode === "batch" ? "toggle-btn active" : "toggle-btn"}
          onClick={() => setMode("batch")}
        >Batch Analysis</button>
      </div>

      {mode === "single" && (
        <div className="card">
          <textarea
            className="textarea"
            rows={5}
            placeholder="Paste a company review here...&#10;e.g. Great culture and benefits but terrible management and no work life balance"
            value={text}
            onChange={e => setText(e.target.value)}
          />
          <div className="char-count">{text.length} / 1000 characters</div>
          <button
            className="analyze-btn"
            onClick={analyzeSingle}
            disabled={loading || !text.trim()}
          >
            {loading ? "Analyzing..." : "Analyze Sentiment"}
          </button>

          {error && <div className="error">{error}</div>}

          {result && (
            <div className="result-card">
              <div className="result-header">
                <span className="emoji">{getSentimentEmoji(result.sentiment)}</span>
                <span
                  className="sentiment-label"
                  style={{ color: getSentimentColor(result.sentiment) }}
                >
                  {result.sentiment}
                </span>
                <span className="confidence">({result.confidence_pct} confidence)</span>
              </div>

              {result.is_mixed && result.parts_analyzed && (
                <div className="mixed-notice">
                  ⚠️ Mixed signals detected — analyzed in 2 parts
                  <div className="parts">
                    {result.parts_analyzed.map((p, i) => (
                      <div key={i} className="part-chip">
                        <span className="part-emoji">{getSentimentEmoji(p.sentiment)}</span>
                        <span className="part-text">"{p.text}"</span>
                        <span className="part-sentiment" style={{ color: getSentimentColor(p.sentiment) }}>
                          {p.sentiment} · {p.confidence_pct}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="scores-section">
                <p className="scores-title">Confidence breakdown</p>
                {result.all_scores && Object.entries(result.all_scores).map(([label, val]) => (
                  <ScoreBar key={label} label={label} value={val} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {mode === "batch" && (
        <div className="card">
          <p className="batch-hint">Enter one review per line (max 20)</p>
          <textarea
            className="textarea"
            rows={8}
            placeholder={"Great culture and amazing benefits\nToxic management, avoid this company\nDecent pay but no growth opportunities"}
            value={batchText}
            onChange={e => setBatchText(e.target.value)}
          />
          <button
            className="analyze-btn"
            onClick={analyzeBatch}
            disabled={loading || !batchText.trim()}
          >
            {loading ? "Analyzing..." : "Analyze Batch"}
          </button>

          {error && <div className="error">{error}</div>}

          {batchResults && (
            <div className="batch-results">
              <div className="summary">
                <span className="sum-chip positive">✅ Positive: {batchResults.summary.Positive || 0}</span>
                <span className="sum-chip neutral">➖ Neutral: {batchResults.summary.Neutral || 0}</span>
                <span className="sum-chip negative">🚨 Negative: {batchResults.summary.Negative || 0}</span>
                <span className="sum-chip mixed">⚠️ Mixed: {batchResults.summary.Mixed || 0}</span>
              </div>
              {batchResults.results.map((r, i) => (
                <div key={i} className="batch-item">
                  <div className="batch-text">"{r.text}"</div>
                  <div className="batch-sentiment" style={{ color: getSentimentColor(r.sentiment) }}>
                    {getSentimentEmoji(r.sentiment)} {r.sentiment} · {r.confidence_pct}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <footer className="footer">
        Model: <a href="https://huggingface.co/Madhuri1003/workpulse-distilbert" target="_blank" rel="noreferrer">
          Madhuri1003/workpulse-distilbert
        </a> · Dataset: 838K Glassdoor Reviews
      </footer>
    </div>
  );
}