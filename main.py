from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from complaint_agent import analyze_reviews, predict_sentiment

# ============================================================
# APP SETUP
# ============================================================
app = FastAPI(
    title="AI Complaint Intelligence Agent",
    description="Analyzes customer reviews using BERT + CrewAI multi-agent system",
    version="1.0.0"
)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================
class Review(BaseModel):
    text: str
    category: str

class ReviewBatch(BaseModel):
    reviews: List[Review]

class SentimentResponse(BaseModel):
    text: str
    category: str
    sentiment: str
    confidence: float

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "name": "AI Complaint Intelligence Agent",
        "status": "running",
        "endpoints": ["/predict", "/analyze", "/health"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=SentimentResponse)
def predict(review: Review):
    """
    Single review sentiment prediction using BERT.
    Fast — no agents involved.
    """
    try:
        sentiment, confidence = predict_sentiment(review.text, review.category)
        return SentimentResponse(
            text=review.text,
            category=review.category,
            sentiment=sentiment,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze(batch: ReviewBatch):
    """
    Full multi-agent analysis on a batch of reviews.
    Runs BERT + all 4 CrewAI agents.
    Returns complete business report.
    """
    try:
        reviews = [{"text": r.text, "category": r.category} for r in batch.reviews]
        report = analyze_reviews(reviews)
        return {
            "status": "success",
            "review_count": len(reviews),
            "report": str(report)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)