# Transformer-Based Text Summarization & Evaluation System

A deep learning text summarization and comparative evaluation system built with pre-trained Transformer models (**BART** & **T5**), graph-based extractive summarization (**TextRank**, **Lead-3**), and **ROUGE** metric benchmarking (**ROUGE-1, ROUGE-2, ROUGE-L**).

---

## 🌟 Key Features

1. **Pre-trained Transformer Models**:
   - **BART** (`sshleifer/distilbart-cnn-12-6` / `facebook/bart-large-cnn`): Bidirectional and Auto-Regressive Transformer fine-tuned on the CNN/Daily Mail dataset.
   - **T5** (`t5-small` / `t5-base`): Text-to-Text Transfer Transformer with sequence-to-sequence conditional text generation.
2. **Long Text Handling (Hierarchical Map-Reduce)**:
   - Overcomes strict token limits (1024 tokens for BART, 512 tokens for T5).
   - Splits long documents along sentence boundaries with overlapping context.
   - Summarizes chunks in parallel (Map) and condenses them into a final coherent summary (Reduce).
3. **Comprehensive Extractive vs. Abstractive Comparison**:
   - **Extractive**: TextRank (PageRank on sentence TF-IDF cosine graph) & Lead-3 baseline.
   - **Abstractive**: BART and T5 seq2seq generation.
   - Evaluates word counts, compression ratios, latency, and grammatical fluency.
4. **Full ROUGE Metric Suite**:
   - **ROUGE-1**: Unigram overlap (lexical coverage).
   - **ROUGE-2**: Bigram overlap (phrasing and fluency).
   - **ROUGE-L / ROUGE-Lsum**: Longest Common Subsequence (structural coherence).
   - Reports **Precision**, **Recall**, and **F1-Score**.
5. **Datasets**:
   - **CNN/Daily Mail (v3.0.0)** & News Article benchmarks with human gold-standard summaries.
   - Streaming support via Hugging Face `datasets` library.
6. **Interfaces**:
   - **Web UI**: Interactive dashboard with real-time summarizer, side-by-side comparison, and ROUGE tables.
   - **CLI Tool**: `interactive_cli.py` for terminal usage and file summarization.
   - **Experiment Runner**: `run_experiments.py` for automated benchmark evaluation.

---

## 📁 Project Structure

```
e:\NLP/
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── loader.py              # CNN/Daily Mail & news article loader with streaming & benchmark sets
│   ├── summarizer/
│   │   ├── __init__.py
│   │   ├── chunker.py             # Sentence-boundary text chunker for long documents
│   │   ├── abstractive.py         # Transformer summarizer (BART & T5) with map-reduce
│   │   └── extractive.py          # Extractive summarizers (TextRank, Lead-3, TF-IDF)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── rouge_eval.py          # ROUGE-1, ROUGE-2, ROUGE-L precision/recall/F1 evaluator
│   └── pipeline.py                # Unified end-to-end comparative pipeline
├── app/
│   ├── server.py                  # FastAPI backend server
│   └── static/                    # Modern web dashboard (HTML5, Vanilla CSS, JS)
│       ├── index.html
│       ├── style.css
│       └── app.js
├── results/                       # Generated benchmark evaluations and comparison tables
├── run_experiments.py             # Benchmark runner on CNN/Daily Mail test samples
├── interactive_cli.py             # CLI summarization tool
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Run Automated Evaluation Benchmark
Runs Extractive (Lead-3, TextRank) and Abstractive (BART, T5) models on CNN/Daily Mail benchmark samples, computes ROUGE-1, ROUGE-2, and ROUGE-L, and prints comparative tables:

```bash
python run_experiments.py
```

Results are saved to:
- `results/benchmark_results.json`
- `results/summary_comparison.md`

### 2. Interactive CLI Summarization
Summarize any text file or test dataset samples from the terminal:

```bash
# Summarize a benchmark sample (1-5)
python interactive_cli.py --sample 1

# Summarize a custom document file
python interactive_cli.py --file my_article.txt --model sshleifer/distilbart-cnn-12-6
```

### 3. Launch Web Dashboard
Start the FastAPI server and open the interactive UI in your browser:

```bash
python app/server.py
```
Open `http://127.0.0.1:8000` in your web browser.

---

## 📊 Extractive vs. Abstractive Comparison

| Feature | Extractive (TextRank / Lead-3) | Abstractive (BART / T5) |
| :--- | :--- | :--- |
| **Mechanism** | Extracts salient verbatim sentences from original document | Generates new paraphrased text via sequence-to-sequence transformer |
| **Grammatical Flow** | Can feel disconnected between extracted sentences | Coherent, human-like paraphrasing and narrative flow |
| **Factual Consistency** | Always faithful to source (exact quotes) | Risk of hallucination without proper beam search constraints |
| **Compute / Speed** | Extremely fast (milliseconds on CPU) | Requires neural network forward passes (seconds on CPU, fast on GPU) |
| **ROUGE Performance** | High recall due to full sentence extraction | High precision and concise summaries matching human targets |
