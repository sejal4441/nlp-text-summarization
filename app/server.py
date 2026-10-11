import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

# Ensure workspace root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset.loader import DatasetLoader, BENCHMARK_NEWS_SAMPLES
from src.pipeline import SummarizationPipeline
from src.evaluation.rouge_eval import RougeEvaluator

app = FastAPI(title="Transformer Text Summarization & Evaluation System")

# Global pipeline instance
pipeline = SummarizationPipeline(
    abstractive_model_name="sshleifer/distilbart-cnn-12-6",
    enable_t5=True,
    t5_model_name="t5-small"
)
rouge_evaluator = RougeEvaluator()

class SummarizeRequest(BaseModel):
    text: str
    reference: Optional[str] = None
    model_choice: str = "bart" # "bart", "t5", or "both"
    extractive_method: str = "textrank" # "textrank", "lead_k", "both"
    max_length: int = 142
    min_length: int = 35
    extractive_sentences: int = 3

@app.get("/api/benchmark-samples")
def get_samples():
    """Returns curated CNN/Daily Mail and News Article benchmark samples."""
    return BENCHMARK_NEWS_SAMPLES

@app.post("/api/summarize")
def summarize_document(req: SummarizeRequest):
    """
    Summarize a text document using chosen Transformer and Extractive models,
    and compute ROUGE-1, ROUGE-2, ROUGE-L if reference is provided.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Input document text cannot be empty.")

    pipeline.load_models()
    orig_words = len(req.text.split())

    summaries = {}
    latencies = {}

    # Extractive summarizers
    import time
    if req.extractive_method in ["lead_k", "both"]:
        t0 = time.time()
        s_lead = pipeline.extractive_summarizer.summarize_lead_k(req.text, k=req.extractive_sentences)
        t_lead = round(time.time() - t0, 4)
        summaries["Extractive (Lead-3)"] = {
            "text": s_lead,
            "type": "extractive",
            "words": len(s_lead.split()),
            "compression": round(len(s_lead.split()) / max(1, orig_words), 3),
            "latency": t_lead
        }

    if req.extractive_method in ["textrank", "both"]:
        t0 = time.time()
        s_tr = pipeline.extractive_summarizer.summarize_textrank(req.text, num_sentences=req.extractive_sentences)
        t_tr = round(time.time() - t0, 4)
        summaries["Extractive (TextRank)"] = {
            "text": s_tr,
            "type": "extractive",
            "words": len(s_tr.split()),
            "compression": round(len(s_tr.split()) / max(1, orig_words), 3),
            "latency": t_tr
        }

    # Abstractive models
    if req.model_choice in ["bart", "both"] and pipeline.bart_summarizer is not None:
        t0 = time.time()
        s_bart = pipeline.bart_summarizer.summarize(
            req.text,
            min_length=req.min_length,
            max_length=req.max_length,
            hierarchical_long_text=True
        )
        t_bart = round(time.time() - t0, 3)
        summaries["Abstractive (BART / DistilBART)"] = {
            "text": s_bart,
            "type": "abstractive",
            "words": len(s_bart.split()),
            "compression": round(len(s_bart.split()) / max(1, orig_words), 3),
            "latency": t_bart
        }

    if req.model_choice in ["t5", "both"] and pipeline.t5_summarizer is not None:
        t0 = time.time()
        s_t5 = pipeline.t5_summarizer.summarize(
            req.text,
            min_length=req.min_length,
            max_length=req.max_length,
            hierarchical_long_text=True
        )
        t_t5 = round(time.time() - t0, 3)
        summaries["Abstractive (T5-Small)"] = {
            "text": s_t5,
            "type": "abstractive",
            "words": len(s_t5.split()),
            "compression": round(len(s_t5.split()) / max(1, orig_words), 3),
            "latency": t_t5
        }

    # Evaluation
    evaluation = {}
    if req.reference and req.reference.strip():
        for name, data in summaries.items():
            evaluation[name] = rouge_evaluator.evaluate_single(
                reference=req.reference,
                prediction=data["text"]
            )

    return {
        "document_stats": {
            "word_count": orig_words,
            "char_count": len(req.text)
        },
        "summaries": summaries,
        "evaluation": evaluation,
        "reference": req.reference
    }

@app.get("/api/benchmark-results")
def get_benchmark_results():
    results_path = root_dir / "results" / "benchmark_results.json"
    if results_path.exists():
        import json
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "No benchmark results computed yet. Run run_experiments.py to generate."}

# Mount static files
static_dir = Path(__file__).resolve().parent / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
