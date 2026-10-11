import sys
import argparse
from src.dataset.loader import DatasetLoader
from src.pipeline import SummarizationPipeline
from src.evaluation.rouge_eval import RougeEvaluator

def main():
    parser = argparse.ArgumentParser(description="Transformer Text Summarization & Comparison CLI")
    parser.add_argument("--model", type=str, default="sshleifer/distilbart-cnn-12-6", help="Abstractive model (BART or T5)")
    parser.add_argument("--file", type=str, help="Path to input text file to summarize")
    parser.add_argument("--sample", type=int, default=1, help="Sample index from CNN/Daily Mail benchmark (1-5)")
    args = parser.parse_args()

    pipeline = SummarizationPipeline(
        abstractive_model_name=args.model,
        enable_t5="t5" in args.model.lower()
    )

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            article = f.read()
        reference = None
        title = args.file
    else:
        samples = DatasetLoader.get_benchmark_samples()
        sample = samples[max(0, min(args.sample - 1, len(samples) - 1))]
        article = sample["article"]
        reference = sample["highlights"]
        title = f"[{sample['dataset']}] {sample['title']}"

    print("\n" + "=" * 70)
    print("TEXT SUMMARIZATION COMPARATIVE SYSTEM")
    print("=" * 70)
    print(f"Title: {title}")
    print(f"Original Text ({len(article.split())} words):\n{article[:350]}...\n")

    if reference:
        print(f"Reference Summary (Ground Truth):\n{reference}\n")

    print("Generating Summaries (Extractive: Lead-3, TextRank | Abstractive: Transformer)...")
    res = pipeline.summarize_and_compare(article, reference_summary=reference)

    print("\n" + "-" * 70)
    print("GENERATED SUMMARIES")
    print("-" * 70)
    for method, dat in res["summaries"].items():
        print(f"\n>>> {method} <<<")
        print(f"Length: {dat['word_count']} words | Compression: {dat['compression_ratio']*100:.1f}% | Time: {dat['latency_seconds']}s")
        print(f"Summary: {dat['text']}")

    if res["evaluation"]:
        print("\n" + "=" * 70)
        print("ROUGE EVALUATION METRICS (Precision / Recall / F1)")
        print("=" * 70)
        evaluator = RougeEvaluator()
        table = evaluator.format_scores_table(res["evaluation"])
        print(table)
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
