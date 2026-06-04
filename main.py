from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
from typing import List
import os
from dotenv import load_dotenv
import time

load_dotenv()

app = FastAPI(
    title="WorkPulse API",
    description="Company culture sentiment analysis using fine-tuned DistilBERT on 838K Glassdoor reviews",
    version="1.0.0"
)

# CORS — allows React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — not on every request
MODEL_NAME = os.getenv("HF_MODEL", "Madhuri1003/workpulse-distilbert")
classifier = None

@app.on_event("startup")
async def load_model():
    global classifier
    print(f"Loading model: {MODEL_NAME}")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=-1  # CPU for deployment; change to 0 if GPU available
    )
    print("Model loaded successfully.")

# --- Request/Response schemas ---

class ReviewRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Great culture, supportive management, excellent work life balance"
            }
        }

class BatchReviewRequest(BaseModel):
    texts: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Amazing team and great benefits",
                    "Toxic management, no work life balance",
                    "Average company, nothing special"
                ]
            }
        }

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    confidence_pct: str

class BatchSentimentResult(BaseModel):
    results: List[SentimentResult]
    total: int
    summary: dict

# --- Endpoints ---

@app.get("/")
def root():
    return {
        "name": "WorkPulse API",
        "description": "Company culture sentiment analysis",
        "model": MODEL_NAME,
        "endpoints": ["/predict", "/batch", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model": MODEL_NAME
    }

@app.post("/predict")
def predict(request: ReviewRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long, max 1000 characters")

    # Detect contrast keywords — split and analyze separately
    contrast_keywords = [" but ", " however ", " although ", " yet ", " though ", " despite "]
    parts = [text]
    for kw in contrast_keywords:
        if kw in text.lower():
            idx = text.lower().find(kw)
            parts = [text[:idx].strip(), text[idx+len(kw):].strip()]
            parts = [p for p in parts if len(p.split()) >= 3]  # ignore tiny fragments
            break

    is_mixed = len(parts) > 1

    # Score each part
    all_results = []
    for part in parts:
        result = classifier(part, top_k=3)
        scores = {r["label"]: round(r["score"], 4) for r in result}
        all_results.append(scores)

    # Average scores across parts
    final_scores = {
        "Negative": round(sum(r.get("Negative", 0) for r in all_results) / len(all_results), 4),
        "Neutral":  round(sum(r.get("Neutral", 0)  for r in all_results) / len(all_results), 4),
        "Positive": round(sum(r.get("Positive", 0) for r in all_results) / len(all_results), 4),
    }

    top_label = max(final_scores, key=final_scores.get)
    top_score = final_scores[top_label]

    # If top two scores are within 20% of each other → mixed
    sorted_scores = sorted(final_scores.values(), reverse=True)
    is_mixed = is_mixed or (sorted_scores[0] - sorted_scores[1] < 0.20)

    return {
        "text": text,
        "sentiment": "Mixed" if is_mixed else top_label,
        "confidence": top_score,
        "confidence_pct": f"{top_score*100:.1f}%",
        "is_mixed": is_mixed,
        "all_scores": final_scores,
        "parts_analyzed": parts if is_mixed else None
    }

@app.post("/batch", response_model=BatchSentimentResult)
def batch_predict(request: BatchReviewRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(request.texts) > 20:
        raise HTTPException(status_code=400, detail="Max 20 texts per batch")

    texts = [t.strip() for t in request.texts if t.strip()]
    results_raw = classifier(texts)

    results = []
    summary = {"Positive": 0, "Neutral": 0, "Negative": 0}

    for text, result in zip(texts, results_raw):
        sentiment = result["label"]
        summary[sentiment] = summary.get(sentiment, 0) + 1
        results.append(SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=round(result["score"], 4),
            confidence_pct=f"{result['score']*100:.1f}%"
        ))

    return BatchSentimentResult(
        results=results,
        total=len(results),
        summary=summary
    )