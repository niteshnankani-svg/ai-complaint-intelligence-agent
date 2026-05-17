# AI Complaint Intelligence Agent

> 7 fine-tuned BERT classifiers + 4-agent CrewAI pipeline that turns 51K customer complaints into actionable business intelligence.

## Problem Statement
E-commerce operations teams are buried in unstructured customer complaints — knowing *how many* complaints exist is useless without knowing *which categories spike, which are urgent, and what the trend looks like*. Manual triage at scale is impossible.

## Architecture
Stage 1: 7 category-specific BERT models (fine-tuned on 51K balanced Flipkart reviews) classify each complaint into its domain (delivery, product quality, payment, etc.). Stage 2: A 4-agent CrewAI pipeline processes the classified output — Agent 1 (Complaint Classifier) routes complaints; Agent 2 (Priority Ranker) scores urgency; Agent 3 (Trend Detector) identifies spikes across time windows; Agent 4 (Report Generator) produces an executive summary. FastAPI serves the pipeline; Gradio renders the report UI.

## Tech Stack
`Python` · `BERT (HuggingFace Transformers)` · `CrewAI` · `FastAPI` · `Gradio` · `scikit-learn` · `pandas` · `HuggingFace Spaces`

## Key Results
- 95–99% classification accuracy across all 7 BERT models
- 51,000 training samples (balanced per category)
- Full pipeline: complaint in → structured intelligence report out in <10s
- Deployed on HuggingFace Spaces; live

## Live Demo
🔗 [huggingface.co/spaces/nitz0219/ai-complaint-intelligence-agent](https://huggingface.co/spaces/nitz0219/ai-complaint-intelligence-agent)

## How to Run Locally
```bash
git clone https://github.com/niteshnankani-svg/ai-complaint-intelligence-agent
cd ai-complaint-intelligence-agent
python -m venv venv && source venv/bin/activate   # use Python 3.11
pip install -r requirements.txt
uvicorn app:app --reload
# Open http://localhost:8000
```
