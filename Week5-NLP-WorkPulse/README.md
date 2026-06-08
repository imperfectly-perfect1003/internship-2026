# WorkPulse — Company Culture Sentiment Analyzer
![CI](https://github.com/imperfectly-perfect1003/internship-2026/actions/workflows/ci.yml/badge.svg)
![HuggingFace](https://img.shields.io/badge/🤗%20Model-Madhuri1003/workpulse--distilbert-yellow)
![Vercel](https://img.shields.io/badge/Frontend-Vercel-black)
![HF Spaces](https://img.shields.io/badge/Backend-HuggingFace%20Spaces-blue)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![React](https://img.shields.io/badge/React-18-61dafb)

📐 [Architecture Diagram](./architecture.md)

> Fine-tuned DistilBERT on 838K real Glassdoor reviews to classify company culture as Positive, Negative, Neutral, or Mixed.

## 🔗 Live Links
- **Demo:** https://workpulse-frontend-pied.vercel.app
- **API:** https://madhuri1003-workpulse-api.hf.space/docs
- **Model:** https://huggingface.co/Madhuri1003/workpulse-distilbert

---

## Problem Statement
Job seekers and recruiters need a quick way to understand company culture from employee reviews. Manually reading hundreds of Glassdoor reviews is time-consuming. WorkPulse automates this using NLP.

---

## Dataset
- **Source:** Gopinath-AI/glassdoor_reviews (HuggingFace)
- **Size:** 838,566 real Glassdoor employee reviews
- **Columns used:** `pros`, `cons`, `overall_rating`

### Key Data Decision
During EDA, we discovered that **46.2% of negative reviews (rating 1-2) contained positive language in their `pros` field**. Combining `pros + cons` as input would have created label-text mismatch — the model would learn that positive language = negative sentiment.

**Solution:** Use text that matches the label:
- Negative (rating 1-2) → `cons` text only
- Neutral (rating 3) → `cons` text only  
- Positive (rating 4-5) → `pros` text only

This was validated by keyword analysis before training.

---

## Model

| Component | Detail |
|---|---|
| Base model | distilbert-base-uncased |
| Task | 3-class text classification |
| Training samples | 50,000 (balanced: 16,666 per class) |
| Epochs | 3 |
| Batch size | 32 |
| Max token length | 128 |

### Training Results

| Metric | Score |
|---|---|
| Overall Accuracy | 78.9% |
| F1 — Negative | 0.689 |
| F1 — Neutral | 0.712 |
| F1 — Positive | 0.966 |
| F1 — Macro avg | 0.789 |

**Honest assessment:** Positive class performs strongest because pros text has cleaner signal. Neutral class is weakest due to inherently ambiguous language in rating-3 reviews.

---

## Mixed Sentiment Detection
Single-label classification cannot explicitly detect mixed sentiment. We implemented a practical workaround:

1. Detect contrast keywords (but, however, although, yet, though, despite)
2. Split text at contrast point and score each part separately
3. Average scores across parts
4. If top two scores differ by less than 20%, classify as Mixed

Example:Input: "Great salary and smart colleagues but management is toxic"
Part 1: "Great salary and smart colleagues" → Positive 94%
Part 2: "management is toxic" → Negative 91%
Result: Mixed (Positive 49.9%, Negative 48.1%)
---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/predict` | POST | Analyze single review |
| `/batch` | POST | Analyze up to 20 reviews |

### Example Request
```bash
curl -X POST https://madhuri1003-workpulse-api.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great culture but terrible management"}'
```

### Example Response
```json
{
  "sentiment": "Mixed",
  "confidence": 0.50,
  "confidence_pct": "50.0%",
  "is_mixed": true,
  "all_scores": {
    "Negative": 0.48,
    "Neutral": 0.02,
    "Positive": 0.50
  }
}
```

---

## Tech Stack
- **Model:** HuggingFace Transformers (DistilBERT)
- **Training:** PyTorch + HuggingFace Trainer API
- **API:** FastAPI + Uvicorn
- **Deployment:** HuggingFace Spaces (Docker)
- **Dataset:** HuggingFace Datasets

---

## Limitations
- Neutral class F1 of 0.71 — ambiguous reviews are sometimes misclassified
- Mixed detection relies on contrast keywords — misses implicit mixed sentiment
- Model trained on English reviews only
- 50K training samples used (838K available) due to compute constraints

## Future Work
- Aspect-based sentiment analysis (Management, Culture, Compensation, Work-Life Balance separately) — dataset has per-aspect ratings available
- Temperature scaling for confidence calibration
- Company-level culture report cards by aggregating multiple reviews
