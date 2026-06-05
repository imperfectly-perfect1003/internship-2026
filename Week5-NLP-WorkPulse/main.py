from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
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

HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_URL = "https://api-inference.huggingface.co/models/Madhuri1003/workpulse-distilbert"

# --- Schemas ---
class ReviewRequest(BaseModel):
    text: str

class BatchReviewRequest(BaseModel):
    texts: List[str]

# --- HuggingFace Inference call ---
async def hf_predict(texts: List[str]):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            MODEL_URL,
            headers=headers,
            json={"inputs": texts, "parameters": {"top_k": 3}}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HuggingFace API error: {response.text}")
        return response.json()

def parse_scores(raw):
    scores = {item["label"]: round(item["score"], 4) for item in raw}
    return scores

def detect_mixed(text, scores):
    contrast_keywords = [" but ", " however ", " although ", " yet ", " though ", " despite "]
    is_split = any(kw in text.lower() for kw in contrast_keywords)
    sorted_scores = sorted(scores.values(), reverse=True)
    is_close = (sorted_scores[0] - sorted_scores[1]) < 0.20
    return is_split or is_close   
# --- Endpoints ---
@app.get("/")
def root():
    return {"name": "WorkPulse API", "model": "Madhuri1003/workpulse-distilbert"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(request: ReviewRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(text) > 1000:
        raise HTTPException(status_code=400, detail="Text too long")

    result = await hf_predict([text])
    # HF returns list of list for batch, list for single
    raw = result[0] if isinstance(result[0], list) else result
    scores = parse_scores(raw)
    top_label = max(scores, key=scores.get)
    top_score = scores[top_label]
    is_mixed = detect_mixed(text, scores)
    return {
        "text": text,
        "sentiment": "Mixed" if is_mixed else top_label,
        "confidence": top_score,
        "confidence_pct": f"{top_score*100:.1f}%",
        "is_mixed": is_mixed,
        "all_scores": scores,
        "parts_analyzed": None
    }

@app.post("/batch")
async def batch_predict(request: BatchReviewRequest):
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    if len(request.texts) > 20:
        raise HTTPException(status_code=400, detail="Max 20 texts per batch")

    texts = [t.strip() for t in request.texts if t.strip()]
    result = await hf_predict(texts)
    results = []
    summary = {"Positive": 0, "Neutral": 0, "Negative": 0, "Mixed": 0}

    for text, raw in zip(texts, result):
        scores = parse_scores(raw if isinstance(raw, list) else result)
        top_label = max(scores, key=scores.get)
        top_score = scores[top_label]
        is_mixed = detect_mixed(text, scores)
        sentiment = "Mixed" if is_mixed else top_label
        summary[sentiment] = summary.get(sentiment, 0) + 1
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": top_score,
            "confidence_pct": f"{top_score*100:.1f}%"
        })

    return {"results": results, "total": len(results), "summary": summary}