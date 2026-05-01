import gradio as gr
import pandas as pd
import json
import os
from complaint_agent import analyze_reviews, predict_sentiment

CATEGORIES = [
    "Electronics",
    "Appliances",
    "Fashion",
    "Home",
    "Kitchen",
    "General"
]

def predict_single(review_text, category):
    if not review_text.strip():
        return "Please enter a review.", ""
    try:
        sentiment, confidence = predict_sentiment(review_text, category)
        sentiment = sentiment.upper()
        confidence = confidence * 100
        emoji = "✅" if sentiment == "POSITIVE" else "❌"
        return f"{emoji} {sentiment}", f"Confidence: {confidence:.1f}%"
    except Exception as e:
        return f"Error: {str(e)}", ""

def analyze_batch(reviews_text):
    if not reviews_text.strip():
        return "Please enter reviews."
    lines = [l.strip() for l in reviews_text.strip().split('\n') if l.strip()]
    reviews = []
    for line in lines:
        if ':' in line:
            category, text = line.split(':', 1)
            category = category.strip()
            text = text.strip()
            if category in CATEGORIES:
                reviews.append({"text": text, "category": category})
            else:
                reviews.append({"text": line, "category": "General"})
        else:
            reviews.append({"text": line, "category": "General"})
    if not reviews:
        return "No valid reviews found."
    try:
        report = analyze_reviews(reviews)
        return str(report)
    except Exception as e:
        return f"Error: {str(e)}"

def extract_category_from_name(product_name):
    name_lower = str(product_name).lower()
    keywords = {
        'Electronics': ['phone', 'mobile', 'laptop', 'tablet', 'camera', 'speaker', 'headphone', 'tv', 'monitor'],
        'Appliances': ['cooler', 'fan', 'ac', 'refrigerator', 'washing', 'microwave', 'mixer', 'iron'],
        'Fashion': ['shirt', 'jeans', 'dress', 'saree', 'kurta', 'shoes', 'sandal', 'watch', 'bag'],
        'Beauty': ['cream', 'lotion', 'lipstick', 'shampoo', 'soap', 'perfume', 'moisturizer'],
        'Home': ['bed', 'sofa', 'chair', 'table', 'curtain', 'pillow', 'mattress', 'lamp'],
        'Kitchen': ['pan', 'pot', 'cooker', 'bottle', 'knife', 'spoon', 'plate', 'tiffin'],
    }
    for category, words in keywords.items():
        for word in words:
            if word in name_lower:
                return category
    return 'General'

def analyze_file(file, text_col, category_col):
    if file is None:
        return "Please upload a file.", ""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        else:
            df = pd.read_excel(file.name)
        summary = f"File loaded: {len(df):,} rows, {len(df.columns)} columns\n"
        summary += f"Columns: {df.columns.tolist()}\n"
        if text_col not in df.columns:
            return f"Column '{text_col}' not found. Available: {df.columns.tolist()}", summary
        if category_col in df.columns:
            df['_category'] = df[category_col].astype(str)
        elif 'product_name' in df.columns:
            df['_category'] = df['product_name'].apply(extract_category_from_name)
        else:
            df['_category'] = 'General'
        df = df.dropna(subset=[text_col])
        summary += f"Reviews after cleaning: {len(df):,}"
        if len(df) > 50:
            df = df.head(50)
            summary += f"\n(Processing first 50 reviews)"
        reviews = []
        for _, row in df.iterrows():
            reviews.append({
                'text': str(row[text_col]),
                'category': str(row['_category'])
            })
        report = analyze_reviews(reviews)
        return str(report), summary
    except Exception as e:
        return f"Error: {str(e)}", ""

with gr.Blocks(title="AI Complaint Intelligence Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI Complaint Intelligence Agent
    ### Powered by BERT + CrewAI Multi-Agent System
    Analyze customer reviews with category-specific BERT models and 4 specialized AI agents.
    """)

    with gr.Tabs():
        with gr.Tab("🔍 Single Review Analysis"):
            gr.Markdown("### Instant sentiment prediction using fine-tuned BERT")
            with gr.Row():
                with gr.Column():
                    review_input = gr.Textbox(label="Customer Review", placeholder="Enter a customer review here...", lines=4)
                    category_input = gr.Dropdown(choices=CATEGORIES, value="Electronics", label="Product Category")
                    predict_btn = gr.Button("Analyze Sentiment", variant="primary")
                with gr.Column():
                    sentiment_output = gr.Textbox(label="Sentiment", lines=1)
                    confidence_output = gr.Textbox(label="Confidence", lines=1)
            predict_btn.click(fn=predict_single, inputs=[review_input, category_input], outputs=[sentiment_output, confidence_output])
            gr.Examples(
                examples=[
                    ["Battery completely dead after 2 days.", "Electronics"],
                    ["Amazing product, works perfectly!", "Appliances"],
                    ["Poor stitching, came apart after first wash.", "Fashion"],
                    ["Fast delivery, great packaging.", "Kitchen"],
                ],
                inputs=[review_input, category_input]
            )

        with gr.Tab("📊 Multi-Agent Batch Analysis"):
            gr.Markdown("""
            ### Full CrewAI agent pipeline
            Enter one review per line: `Category: Review text`
            """)
            with gr.Row():
                with gr.Column():
                    batch_input = gr.Textbox(
                        label="Customer Reviews (one per line)",
                        placeholder="Electronics: Battery dead after 2 days\nAppliances: Product never arrived",
                        lines=8
                    )
                    analyze_btn = gr.Button("🚀 Run Agent Analysis", variant="primary")
                with gr.Column():
                    report_output = gr.Textbox(label="Business Intelligence Report", lines=20)
            analyze_btn.click(fn=analyze_batch, inputs=[batch_input], outputs=[report_output])

        with gr.Tab("📁 Upload CSV / Excel"):
            gr.Markdown("""
            ### Upload your reviews file
            Supports CSV and Excel exported from Amazon, Flipkart, Meesho, Shopify.
            """)
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload CSV or Excel", file_types=[".csv", ".xlsx", ".xls"])
                    text_column = gr.Textbox(label="Review text column name", value="review_text")
                    category_column = gr.Textbox(label="Category column name (optional)", value="category")
                    upload_btn = gr.Button("🚀 Analyze File", variant="primary")
                with gr.Column():
                    file_report_output = gr.Textbox(label="Business Intelligence Report", lines=20)
                    file_summary = gr.Textbox(label="File Summary", lines=5)
            upload_btn.click(fn=analyze_file, inputs=[file_input, text_column, category_column], outputs=[file_report_output, file_summary])

    gr.Markdown("---\nBuilt by **Nitesh Nankani** | BERT + CrewAI + FastAPI + Gradio")

if __name__ == "__main__":
    demo.launch()
