# Swahili ASR & NLP Engineering Pipeline

A production-oriented modular pipeline for **Swahili Automatic Speech Recognition (ASR)** and **textual logical reasoning**, developed for the Speech & NLP / Multimodal ML Engineer practical assessment.

## Results

| Task | Baseline | Final | Improvement |
|---|---:|---:|---:|
| ASR WER | 1.4804 | **0.4298** | **70.97% lower** |
| ASR CER | 0.7637 | **0.1224** | **83.97% lower** |
| Reasoning Exact Match | 0.00 | **0.97** | **97/100 correct** |

### Evaluation
- FLEURS Swahili (`sw_ke`)
- ASR test set: **487 samples**
- Controlled Swahili reasoning test set: **100 samples**
- ASR model: `openai/whisper-small`
- Reasoning model: `Qwen/Qwen2.5-1.5B-Instruct`
- Reasoning adaptation: **QLoRA with assistant-only loss**
- Training hardware: Google Colab Tesla T4, 14.56 GB VRAM

## Architecture

```text
Audio
  │
  ▼
16 kHz preprocessing
  │
  ▼
Whisper Small — Swahili ASR
  │
  ▼
Swahili text normalization
  │
  ▼
Qwen2.5-1.5B-Instruct + QLoRA
  │
  ▼
Short textual reasoning response
```

The implementation intentionally uses a **modular multimodal pipeline** rather than claiming joint training of a single monolithic multimodal model. This was selected because the available compute, data alignment, and task requirements favor specialized components that can be independently evaluated and deployed.

## Repository

```text
Swahili-ASR-NLP-Engineering/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── notebooks/
│   └── Swahili_ASR_NLP_Assessment.ipynb
├── src/
│   ├── data/
│   │   ├── normalization.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── asr.py
│   │   └── reasoning.py
│   ├── evaluation/
│   │   ├── asr_metrics.py
│   │   └── reasoning_metrics.py
│   └── deployment/
│       └── benchmark.py
├── configs/
│   └── config.yaml
├── results/
│   ├── final_asr_results.json
│   ├── final_reasoning_results.json
│   └── final_pipeline_manifest.json
├── reports/
│   └── technical_report.md
├── data/
│   └── README.md
└── models/
    └── README.md
```

## Reproduction

The notebook is designed to be **artifact-first by default** so that opening it does not repeatedly download or decode the full FLEURS dataset.

1. Open the notebook in Google Colab.
2. Enable a GPU for training/inference if reproducing the experiments.
3. Install the pinned dependencies.
4. Set `RUN_EXPENSIVE_EXPERIMENTS = True` only when intentionally rerunning training/evaluation.
5. For normal review, leave it `False` and inspect the saved result artifacts.

### FLEURS compute strategy

FLEURS contains 3,768 Swahili samples and approximately 15.4 hours of audio in the evaluated splits. Audio is decoded lazily/on demand. The full dataset should not be materialized as persistent NumPy arrays in Colab because this can exhaust RAM.

## Model artifacts

Large model checkpoints are **not committed to GitHub**. The repository records the checkpoint locations used in the experiment:

- Whisper fine-tuned checkpoint: `checkpoint-125`
- QLoRA v2 adapter: `checkpoint-100`

The actual weights should be stored in Google Drive, Hugging Face Hub, or another artifact store rather than Git.

## Limitations

The reasoning benchmark is a controlled synthetic 100-example benchmark designed to test rule application, ordering, and arithmetic. The 97% Exact Match result should therefore be interpreted as benchmark-specific rather than as a claim of general reasoning ability.

For ASR, substitution errors were the dominant failure mode in the evaluated test set, with challenges including technical terms, proper nouns, mixed terminology, and phonetic/morphological confusions.

## Technical report

See [`reports/technical_report.md`](reports/technical_report.md) for the detailed methodology, diagnostics, deployment strategy, limitations, and reproducibility notes.

## Task 4 — Edge Deployment & Optimization

Task 4 is implemented as a reproducible deployment plan plus measured GPU benchmarking.
The optimization strategy is deliberately component-specific:

- **Reasoning:** 4-bit NF4 quantization with double quantization and FP16 compute was used during QLoRA training/inference. Only LoRA adapter weights are saved for the adapted component.
- **ASR:** Whisper Small is served in FP16 in the reported benchmark. The measured peak GPU memory was only **3.14 GB** on a 14.56 GB T4, so further ASR quantization was not claimed without an accuracy comparison.
- **Local serving:** checkpoint paths and model composition are represented in `src/deployment/serve.py`.
- **Benchmarking:** `src/deployment/benchmark.py` measures latency, throughput, and peak GPU memory. The recorded benchmark is in `results/edge_benchmark.json`.

### Measured T4 ASR benchmark

| Metric | Measured value |
|---|---:|
| Parameters | 241,734,912 |
| Approx. FP16 weight memory | 0.45 GB |
| Average latency | 1.113 s/sample |
| Minimum latency | 0.810 s |
| Maximum latency | 1.585 s |
| Peak GPU memory | 3.14 GB |

These are **measured T4 results**, not simulated edge-device numbers. For an actual embedded target, the next step is to benchmark INT8/INT4 or ONNX/TensorRT variants on the target hardware and compare WER/CER against the FP16 checkpoint before deployment.
