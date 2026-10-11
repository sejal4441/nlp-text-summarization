import os
import json
import logging
from typing import Dict, Any
from src.dataset.loader import DatasetLoader
from src.pipeline import SummarizationPipeline
from src.evaluation.rouge_eval import RougeEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("experiments")

def run_benchmark(num_samples: int = 3, use_t5: bool = True):
    logger.info("=================================================================")
    logger.info("STARTING TEXT SUMMARIZATION BENCHMARK & COMPARATIVE EVALUATION")
    logger.info("=================================================================")

    # 1. Load Dataset
    loader = DatasetLoader()
    samples = loader.get_benchmark_samples()[:num_samples]
    logger.info(f"Loaded {len(samples)} benchmark evaluation samples (CNN/Daily Mail & News Articles).")

    # 2. Initialize pipeline
    # Using 'sshleifer/distilbart-cnn-12-6' (fast & accurate BART model) and 't5-small'
    pipeline = SummarizationPipeline(
        abstractive_model_name="sshleifer/distilbart-cnn-12-6",
        enable_t5=use_t5,
        t5_model_name="t5-small"
    )
    pipeline.load_models()

    evaluator = RougeEvaluator()

    all_methods = [
        "Extractive (Lead-3)",
        "Extractive (TextRank)",
        "Abstractive (distilbart-cnn-12-6)"
    ]
    if use_t5:
        all_methods.append("Abstractive (t5-small)")

    predictions_by_method = {m: [] for m in all_methods}
    references = []
    latencies_by_method = {m: [] for m in all_methods}
    compression_by_method = {m: [] for m in all_methods}
    sample_records = []

    for idx, sample in enumerate(samples, 1):
        logger.info(f"\n--- Processing Sample {idx}/{len(samples)}: [{sample['dataset']}] {sample['title']} ---")
        art_words = len(sample["article"].split())
        ref_words = len(sample["highlights"].split())
        logger.info(f"Article Word Count: {art_words} | Reference Summary Word Count: {ref_words}")

        comp = pipeline.summarize_and_compare(
            article=sample["article"],
            reference_summary=sample["highlights"]
        )

        references.append(sample["highlights"])
        sample_entry = {
            "id": sample["id"],
            "dataset": sample["dataset"],
            "title": sample["title"],
            "article": sample["article"],
            "reference": sample["highlights"],
            "summaries": {}
        }

        for method in all_methods:
            if method in comp["summaries"]:
                sum_data = comp["summaries"][method]
                predictions_by_method[method].append(sum_data["text"])
                latencies_by_method[method].append(sum_data["latency_seconds"])
                compression_by_method[method].append(sum_data["compression_ratio"])
                sample_entry["summaries"][method] = {
                    "text": sum_data["text"],
                    "word_count": sum_data["word_count"],
                    "latency": sum_data["latency_seconds"],
                    "rouge": comp["evaluation"].get(method, {})
                }
                logger.info(f"  [{method}] Words: {sum_data['word_count']} | Latency: {sum_data['latency_seconds']}s")
                logger.info(f"  Summary Preview: {sum_data['text'][:120]}...")

        sample_records.append(sample_entry)

    # 3. Compute Aggregate Evaluation Metrics
    logger.info("\n=================================================================")
    logger.info("AGGREGATE ROUGE BENCHMARK RESULTS")
    logger.info("=================================================================")

    benchmark_summary = {}
    for method in all_methods:
        mean_rouge = evaluator.evaluate_batch(
            references=references,
            predictions=predictions_by_method[method]
        )
        avg_latency = round(sum(latencies_by_method[method]) / len(latencies_by_method[method]), 3)
        avg_compression = round(sum(compression_by_method[method]) / len(compression_by_method[method]), 3)

        benchmark_summary[method] = {
            **mean_rouge,
            "avg_latency_s": avg_latency,
            "avg_compression_ratio": avg_compression
        }

    # Format into markdown table
    header = (
        "| Summarization Method | Paradigm | ROUGE-1 F1 | ROUGE-1 Rec | ROUGE-2 F1 | ROUGE-2 Rec | ROUGE-L F1 | ROUGE-L Rec | Avg Latency (s) | Compression |"
    )
    separator = "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    table_lines = [header, separator]

    for method, metrics in benchmark_summary.items():
        paradigm = "Abstractive" if "Abstractive" in method else "Extractive"
        r1 = metrics["rouge1"]
        r2 = metrics["rouge2"]
        rl = metrics["rougeL"]
        lat = metrics["avg_latency_s"]
        comp_r = metrics["avg_compression_ratio"]

        row = (
            f"| **{method}** | {paradigm} | "
            f"{r1['fmeasure']:.4f} | {r1['recall']:.4f} | "
            f"{r2['fmeasure']:.4f} | {r2['recall']:.4f} | "
            f"{rl['fmeasure']:.4f} | {rl['recall']:.4f} | "
            f"{lat:.2f}s | {comp_r*100:.1f}% |"
        )
        table_lines.append(row)

    table_markdown = "\n".join(table_lines)
    print("\n" + table_markdown + "\n")

    # 4. Save results to disk
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "aggregate_metrics": benchmark_summary,
            "samples": sample_records
        }, f, indent=2)

    with open("results/summary_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Text Summarization Benchmark Results\n\n")
        f.write("## Overall Performance Comparison (ROUGE-1, ROUGE-2, ROUGE-L)\n\n")
        f.write(table_markdown + "\n\n")
        f.write("## Sample Outputs and Analysis\n\n")
        for rec in sample_records:
            f.write(f"### Sample: {rec['title']} ({rec['dataset']})\n\n")
            f.write(f"**Reference Summary (Ground Truth):**\n> {rec['reference']}\n\n")
            for m, dat in rec["summaries"].items():
                f.write(f"#### {m}\n")
                f.write(f"- **Summary:** {dat['text']}\n")
                f.write(f"- **Words:** {dat['word_count']} | **Latency:** {dat['latency']}s\n")
                r = dat.get("rouge", {})
                if r:
                    f.write(f"- **ROUGE-1 F1:** {r.get('rouge1',{}).get('fmeasure',0.0):.4f} | **ROUGE-2 F1:** {r.get('rouge2',{}).get('fmeasure',0.0):.4f} | **ROUGE-L F1:** {r.get('rougeL',{}).get('fmeasure',0.0):.4f}\n\n")
                else:
                    f.write("\n")

    logger.info("Results successfully saved to 'results/benchmark_results.json' and 'results/summary_comparison.md'")
    return benchmark_summary

if __name__ == "__main__":
    run_benchmark(num_samples=3, use_t5=True)
