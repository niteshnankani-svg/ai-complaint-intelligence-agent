# AI Complaint Intelligence Agent

Turns a batch of raw customer complaints into a structured business-intelligence report: category-specific BERT models score sentiment per complaint, then a 4-agent CrewAI pipeline classifies, prioritizes, and summarizes the negative ones into an executive report.

## Problem statement

E-commerce operations teams are buried in unstructured customer complaints — knowing *how many* complaints exist is less useful than knowing which categories are spiking, which are urgent, and what the trend looks like. This project automates that triage.

## Architecture

**Stage 1 — sentiment classification (`bert_model.py`, `complaint_agent.py`):** one BERT (`bert-base-uncased`) binary sentiment classifier is fine-tuned per product category (Electronics, Appliances, Fashion, Home, Kitchen, and a merged "General" category for categories with too few samples) on the Flipkart product-review dataset, balanced positive/negative per category (`data_loader.py`). At inference time, `complaint_agent.py` loads the matching fine-tuned model per category from the author's HuggingFace Hub repos (`nitz0219/complaint-bert-<category>`) and scores each incoming review.

**Stage 2 — business intelligence (CrewAI, `complaint_agent.py`):** reviews classified as negative are handed to a sequential 4-agent CrewAI pipeline (`gpt-3.5-turbo`): a Complaint Classifier (routes each complaint into delivery/quality/pricing/packaging/customer_service/other), a Priority Ranker (assigns a 1–5 urgency score), a Trend Detector (finds patterns and spikes across the batch), and a Report Generator (writes the executive summary with an overall risk level and recommended actions).

`main.py` / `app.py` expose this as a FastAPI service with a Gradio UI on top.

```
Flipkart reviews (CSV)
        │
data_loader.py → category extraction + balancing
        │
bert_model.py → one fine-tuned BERT per category (train)
        │
complaint_agent.py → load per-category BERT → sentiment + confidence
        │
   [negative reviews only]
        │
CrewAI: Classifier → Priority Ranker → Trend Detector → Report Generator (gpt-3.5-turbo)
        │
   Business intelligence report
```

## Tech stack

Python, BERT (HuggingFace `transformers`, PyTorch), CrewAI, OpenAI `gpt-3.5-turbo`, FastAPI, Gradio, scikit-learn, pandas, HuggingFace Hub (model + Spaces hosting).

## Setup / run

```bash
git clone https://github.com/niteshnankani-svg/ai-complaint-intelligence-agent
cd ai-complaint-intelligence-agent
python -m venv venv && source venv/bin/activate   # Python 3.11
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
uvicorn app:app --reload
# Open http://localhost:8000
```

To retrain the BERT classifiers yourself: place the Flipkart reviews CSV at `data/flipkart_product.csv` (or set `FLIPKART_CSV_PATH`), run `python data_loader.py` to build the balanced training set, then `python bert_model.py` to fine-tune one model per category (writes to `models/`, and a `models/training_results.json` accuracy summary).

## Live demo

[huggingface.co/spaces/nitz0219/ai-complaint-intelligence-agent](https://huggingface.co/spaces/nitz0219/ai-complaint-intelligence-agent)

## Evaluation

Training data: 51,000+ Flipkart reviews, balanced per category (positive/negative). `bert_model.py` computes and prints per-category validation accuracy after each epoch and writes a final `models/training_results.json` — that file (and the trained model weights) are training artifacts, not checked into this repo, so per-category accuracy numbers aren't reproducible from the repo alone; regenerate them by running `bert_model.py` against the dataset.

## Known limitations

- The dataset and trained model weights are not included in this repo (see Setup/run above for how to regenerate them) — only the training and inference code is checked in.
- No automated eval suite or benchmark file is committed; treat published accuracy figures as results from the author's own training run, not something this repo alone reproduces.
- `complaint_agent.py` defaults `OPENAI_API_KEY` to the literal string `'your_key_here'` if unset, which will fail loudly at the OpenAI call rather than at startup — set the real key before running.
