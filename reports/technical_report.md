# Swahili ASR & NLP Engineering Pipeline
## Technical Assessment Report

**Assessment focus:** High-fidelity Swahili speech recognition, textual logical reasoning, evaluation, and edge-oriented deployment.

**Implementation environment:** Google Colab, NVIDIA Tesla T4 (14.56 GB VRAM), Python 3.13.15, PyTorch 2.11.0+cu128.

---

## 1. Executive Summary

This project implements a practical Swahili speech-and-language pipeline that combines a fine-tuned Whisper Small ASR model with a parameter-efficiently fine-tuned Qwen2.5-1.5B-Instruct reasoning model.

The main engineering objective was not simply to select the largest available multimodal model, but to produce a reproducible system that could be trained, evaluated, diagnosed, and benchmarked within a free Google Colab GPU environment.

The resulting system produced substantial measured improvements:

| Component | Baseline | Final | Improvement |
|---|---:|---:|---:|
| ASR WER | 1.4804 | **0.4298** | **70.97% lower** |
| ASR CER | 0.7637 | **0.1224** | **83.97% lower** |
| Reasoning Exact Match | 0.00 | **0.97** | **97/100 correct** |

The ASR result was measured on all 487 FLEURS Swahili test examples. The reasoning result was measured on a 100-example controlled synthetic Swahili reasoning benchmark. The final reasoning model made three errors, all arithmetic errors.

The system is intentionally **modular rather than a monolithic multimodal model**. This decision was driven by compute constraints, task specialization, data availability, evaluation requirements, reproducibility, and deployment considerations. It does not claim joint multimodal parameter training. Instead, the pipeline performs multimodal inference by passing speech through ASR and then passing the resulting text to the reasoning component.

---

# 2. Problem Definition

The target system must handle two related but technically different capabilities:

1. **Speech understanding:** Convert native Swahili speech into accurate written Swahili.
2. **Textual reasoning:** Apply arithmetic, ordering, conditional, and sequence reasoning to Swahili text.

These capabilities have different optimization targets.

For ASR, the central metrics are Word Error Rate (WER) and Character Error Rate (CER). For reasoning, exact-answer correctness is more appropriate for the controlled benchmark.

The implementation therefore separates the two learning problems while integrating them at inference time.

---

# 3. System Architecture

## 3.1 High-level architecture

```text
                   AUDIO INPUT
                       |
                       v
             Audio preprocessing
                       |
                       v
             Whisper Small ASR
             Swahili fine-tuning
                       |
                       v
             Normalized transcript
                       |
                       v
        Qwen2.5-1.5B-Instruct + QLoRA
             Swahili reasoning
                       |
                       v
                  FINAL ANSWER
```

The evaluation architecture is similarly modular:

```text
                 Evaluation Harness
                       |
          +------------+------------+
          |                         |
          v                         v
      ASR branch               NLP branch
          |                         |
     WER / CER                Exact Match
          |                         |
     Error analysis           Error analysis
```

---

# 4. Why a Monolithic Multimodal Model Was Not Used

## 4.1 The decision

A monolithic speech-language multimodal model was considered conceptually, but was not selected for the final implementation.

Instead, the project uses:

- **Whisper Small** for speech recognition.
- **Qwen2.5-1.5B-Instruct + QLoRA** for textual reasoning.
- A lightweight orchestration layer to connect both components.

This is a deliberate engineering decision rather than an omission.

## 4.2 Compute constraint

The experiment was conducted on a free Google Colab Tesla T4 with **14.56 GB VRAM**.

A large multimodal speech-language model would introduce substantially greater memory and training complexity. A monolithic architecture would potentially require:

- larger GPU memory,
- longer training time,
- larger activation memory,
- more aggressive quantization,
- more complex multimodal batching,
- larger model checkpoints,
- and potentially multiple-GPU infrastructure.

Such a setup would not be reliably reproducible in the available environment.

The selected models fit the available hardware much better. The Whisper Small model has approximately **241.7 million parameters**, and its measured benchmark used only **3.14 GB peak GPU memory** during inference on the T4.

## 4.3 Dataset mismatch

The available speech dataset was FLEURS Swahili (`sw_ke`), containing:

- 3,070 training samples,
- 211 validation samples,
- 487 test samples.

The reasoning dataset was a separate controlled synthetic benchmark containing arithmetic, logical ordering, conditional reasoning, and sequence tasks.

These datasets do not constitute a large, aligned speech-to-reasoning multimodal corpus where an end-to-end multimodal model could be trained effectively.

Forcing both tasks into one training objective would therefore create a data-alignment problem rather than solving one.

## 4.4 Task specialization

Speech recognition and reasoning require different learning signals.

ASR learns an acoustic-to-text mapping and is naturally optimized using sequence-generation objectives and evaluated using WER/CER.

Reasoning learns text-to-answer behavior and is evaluated according to answer correctness.

Keeping the components separate makes it possible to:

- optimize ASR independently,
- optimize reasoning independently,
- diagnose errors independently,
- replace one component without retraining the other,
- and select the appropriate metric for each task.

## 4.5 Reproducibility

The assessment was completed under a constrained environment. A modular design makes the experiment easier to reproduce because each component has:

- a clearly defined input,
- a clearly defined output,
- an independent checkpoint,
- independent evaluation,
- and independent resource requirements.

The ASR checkpoint and reasoning adapter can therefore be reused without repeating the complete training pipeline.

## 4.6 Production engineering advantage

A modular architecture also provides operational flexibility.

For example:

- If ASR accuracy improves, the Whisper component can be replaced while keeping the reasoning model.
- If reasoning improves, the Qwen adapter can be replaced without changing the acoustic pipeline.
- ASR and reasoning can be deployed on separate services.
- Each component can be monitored using task-specific metrics.
- Different quantization or latency strategies can be applied independently.

Therefore, the chosen design is better described as a **modular multimodal pipeline**, not as a single multimodal neural network.

---

# 5. Task 1 — Data Engineering and Preprocessing

## 5.1 Dataset

Google FLEURS was loaded using the Swahili Kenya configuration:

```text
google/fleurs
configuration: sw_ke
```

Dataset split:

| Split | Samples |
|---|---:|
| Train | 3,070 |
| Validation | 211 |
| Test | 487 |
| Total | 3,768 |

The audio was standardized around a 16 kHz sampling rate.

## 5.2 Streaming and memory-aware processing

A major engineering issue was memory consumption when attempting to materialize the entire dataset.

Because the full dataset contains thousands of audio examples, loading and decoding all audio arrays simultaneously is inefficient in a constrained Colab environment.

The pipeline therefore uses lazy/on-demand audio decoding for demonstration and processing instead of permanently materializing every waveform in memory.

This is especially important because the final ASR evaluation had already been completed and saved. Reconstructing the entire dataset after a runtime restart was unnecessary and caused avoidable memory pressure.

## 5.3 Audio decoding

The current Hugging Face Datasets audio interface returns an `AudioDecoder`. Audio samples were decoded using:

```python
audio_data = sample["audio"].get_all_samples()
waveform = audio_data.data.squeeze().numpy()
sample_rate = audio_data.sample_rate
```

## 5.4 Text normalization

Swahili transcripts were normalized using Unicode normalization, lowercasing, punctuation removal, and whitespace normalization.

The normalization policy was intentionally conservative and reproducible:

```python
import re
import unicodedata

def normalize_swahili_text(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()
```

Raw transcripts were retained for auditability and qualitative error analysis.

---

# 6. Task 2 — ASR Fine-Tuning

## 6.1 Baseline

The baseline ASR model was:

```text
openai/whisper-small
```

The model was configured for Swahili transcription.

Baseline evaluation on the first test subset produced:

- WER: **1.4804**
- CER: **0.7637**

## 6.2 Fine-tuning strategy

A pilot fine-tuning run used:

- 1,000 training samples,
- 100 validation samples,
- 1 epoch,
- batch size 2,
- gradient accumulation 4,
- learning rate 1e-5,
- FP16,
- generated-sequence evaluation,
- maximum generation length 225.

The run completed in approximately 365 seconds and used 125 optimization steps.

The resulting checkpoint was:

```text
checkpoints/whisper-small-swahili/checkpoint-125
```

## 6.3 Full test-set evaluation

The final model was evaluated on all **487** FLEURS Swahili test samples.

Results:

- WER: **0.4298**
- CER: **0.1224**
- inference time: **10.33 minutes**

Compared with the baseline:

- WER decreased by **70.97%**.
- CER decreased by **83.97%**.

These are measured results rather than estimates.

---

# 7. ASR Error Analysis

The final ASR output contained:

- 9,998 reference words,
- 6,446 correct words,
- 3,176 substitutions,
- 376 deletions,
- 745 insertions.

Error rates:

| Error type | Count | Rate |
|---|---:|---:|
| Substitution | 3,176 | 0.3177 |
| Deletion | 376 | 0.0376 |
| Insertion | 745 | 0.0745 |

Substitution was therefore the dominant error type.

Qualitative inspection showed difficult cases involving:

- technical terminology,
- numerical sequences,
- proper names,
- English/Swahili mixed terminology,
- phonetic confusions,
- morphological confusions,
- and repeated decoding errors in difficult utterances.

This suggests that further gains would likely come from targeted data augmentation, domain-specific speech, better coverage of named entities and technical vocabulary, and more diverse native-speaker recordings.

---

# 8. Task 2 — Textual Reasoning Fine-Tuning

## 8.1 Model

The reasoning component used:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model was adapted using QLoRA.

## 8.2 Why QLoRA

QLoRA was selected because it allows parameter-efficient adaptation of a quantized base model.

The configuration used:

- 4-bit NF4 quantization,
- double quantization,
- FP16 computation,
- LoRA rank 16,
- LoRA alpha 32,
- dropout 0.05,
- target modules covering attention and feed-forward projections,
- learning rate 2e-4,
- batch size 2,
- gradient accumulation 4,
- 2 epochs,
- paged 8-bit AdamW,
- gradient checkpointing.

This reduced the trainable parameter and memory burden compared with full fine-tuning.

---

# 9. Reasoning Benchmark and Training Improvement

The reasoning benchmark was a controlled synthetic Swahili dataset containing several reasoning categories, including:

- arithmetic,
- logical ordering,
- conditional reasoning,
- sequence completion.

For example, arithmetic examples were generated with varying operands rather than simply copying fixed answers.

The first QLoRA variant trained on the full conversational sequence and produced:

- training loss: approximately **0.7366**
- test Exact Match: **0.00**

This demonstrated an important failure mode: a decreasing training loss did not translate into generalization on the held-out reasoning examples.

## 9.1 QLoRA V2

The second variant changed the training objective to **assistant-only loss**.

The prompt tokens were masked with `-100`, so optimization focused on generating the expected answer rather than learning to reproduce the complete prompt.

The instruction was also explicitly constrained to short answers in Swahili.

This substantially improved behavior.

Training:

- 2 epochs,
- 100 steps,
- training loss: **0.3790**,
- runtime: approximately **283.5 seconds**.

Final test performance:

- **97 correct / 100**
- **3 incorrect / 100**
- **Exact Match = 0.97**

The three errors were:

| Sample | Expected | Predicted |
|---|---:|---:|
| 17 | 801 | 701 |
| 72 | 629 | 639 |
| 98 | 222 | 30 |

All three observed errors were arithmetic errors. No logical-ordering or conditional-classification errors were observed in this 100-example benchmark.

The result demonstrates strong improvement on this controlled benchmark, but it should not be interpreted as proof of general-purpose mathematical reasoning.

---

# 10. Task 3 — Evaluation and Diagnostics

## 10.1 ASR metrics

ASR was evaluated using:

- Word Error Rate (WER)
- Character Error Rate (CER)

These metrics capture both word-level and character-level transcription quality.

## 10.2 Reasoning metric

Reasoning was evaluated using normalized Exact Match.

Normalization:

- lowercasing,
- whitespace normalization,
- removal of common surrounding punctuation.

This prevents harmless formatting differences from being counted as reasoning errors.

## 10.3 Multi-task evaluation design

The evaluation harness treats the pipeline as two connected but independently measurable tasks:

```text
Audio
  |
  +--> ASR --> transcript --> WER/CER + error analysis
                          |
                          +--> reasoning component
                                |
                                +--> Exact Match + reasoning error analysis
```

This allows errors to be attributed to the correct stage.

For example, a wrong final answer can arise because:

1. the ASR transcription was incorrect, or
2. the reasoning model misunderstood an otherwise correct transcript.

That separation is valuable for production diagnostics.

---

# 11. Task 4 — Edge Deployment and Benchmarking

The Whisper Small model was benchmarked on the Tesla T4.

Measured model characteristics:

- parameters: **241,734,912**
- approximately **0.24B parameters**
- approximate FP16 weight memory: **0.45 GB**

Measured inference benchmark on 10 samples:

| Metric | Result |
|---|---:|
| Average latency | **1.113 s/sample** |
| Minimum latency | 0.810 s |
| Maximum latency | 1.585 s |
| Peak GPU memory | **3.14 GB** |

The benchmark demonstrates that the ASR component has a relatively manageable inference footprint on the available GPU.

For a production edge deployment, further optimization could include:

- dynamic batching,
- shorter maximum audio windows,
- ONNX/TensorRT conversion where supported,
- lower-precision inference,
- VAD-based segmentation,
- CPU/GPU scheduling,
- model caching,
- and service-level autoscaling.

These additional optimizations were not claimed as measured results in this assessment.

---

# 12. End-to-End Demonstration

A representative FLEURS audio sample was saved as an 18-second, 16 kHz WAV file.

The demonstration successfully passed audio through the ASR stage and produced a transcript.

The observed transcript contained errors such as incorrect years, names, and word segmentation. This is consistent with the error profile observed in the larger ASR evaluation.

Importantly, the end-to-end demonstration is a **pipeline integration demonstration**, not a replacement for the quantitative 487-sample ASR evaluation.

The FLEURS utterance used for demonstration is a transcription-oriented utterance rather than a dedicated reasoning question. Therefore, the report does not incorrectly present it as a multimodal reasoning benchmark.

---

# 13. Production-Oriented Design

The proposed production structure is:

```text
Client
  |
  v
API Gateway
  |
  v
Audio ingestion
  |
  +--> VAD / preprocessing
  |
  v
ASR service
  |
  v
Text normalization
  |
  v
Reasoning service
  |
  v
Response
```

Each service can be independently scaled and monitored.

Recommended monitoring signals include:

### ASR

- WER/CER on labeled samples,
- latency,
- GPU memory,
- audio duration,
- confidence/proxy quality indicators,
- language mismatch,
- OOV/proper-name error frequency.

### Reasoning

- Exact Match,
- task-category accuracy,
- arithmetic error rate,
- latency,
- token generation length,
- malformed-output rate.

### System

- end-to-end latency,
- request throughput,
- GPU utilization,
- memory utilization,
- failure rate,
- queue length.

---

# 14. Limitations

Several limitations should be explicitly acknowledged.

### 14.1 Controlled reasoning dataset

The 100-example reasoning benchmark is synthetic and controlled. A 97% Exact Match result demonstrates strong performance on the tested task distribution, but does not establish broad real-world Swahili reasoning ability.

### 14.2 ASR domain

FLEURS provides a useful multilingual speech benchmark, but it does not fully represent all Tanzanian conversational environments, accents, background noise, code-switching, telephone audio, or domain-specific speech.

### 14.3 Modular rather than jointly trained multimodal model

The system does not jointly update acoustic and language parameters. This was an intentional engineering trade-off caused by the available compute and data conditions.

A future version could investigate a genuine end-to-end multimodal speech-language model once an appropriately aligned speech-plus-reasoning dataset and larger compute budget are available.

### 14.4 Edge benchmark scope

The latency benchmark was conducted on a Tesla T4 and should not be interpreted as an embedded-device benchmark. Real edge deployment would require measurements on the target hardware.

---

# 15. Reproducibility

The experiment stores model checkpoints and quantitative results separately.

Important artifacts include:

```text
checkpoints/
└── whisper-small-swahili/
    └── checkpoint-125/

models/
└── qwen2.5-1.5b-qlora-v2/
    └── checkpoint-100/

results/
└── final_asr_results.json
```

The QLoRA V2 checkpoint contains the adapter configuration and adapter weights required to reconstruct the reasoning component when the corresponding base model is available.

The final ASR result file records the baseline, fine-tuned metrics, error analysis, and inference time.

---

# 16. Key Results

## ASR

The strongest measured result is the reduction of WER from **1.4804 to 0.4298**, representing a **70.97% reduction**, while CER decreased from **0.7637 to 0.1224**, representing an **83.97% reduction**.

## Reasoning

The first QLoRA strategy achieved 0% Exact Match despite a falling training loss. Changing to assistant-only loss and short-answer supervision produced **97% Exact Match** on the same 100-example test set.

This provides an important engineering lesson: **training loss alone is insufficient; the target behavior and loss formulation matter strongly for small instruction-tuning datasets.**

## Deployment

The ASR model achieved **1.113 seconds average latency per sample** with **3.14 GB peak GPU memory** on the Tesla T4 benchmark.

---

# 17. Conclusion

The project demonstrates a practical, reproducible approach to Swahili speech and language processing under constrained compute.

The final pipeline combines:

- memory-aware FLEURS processing,
- Swahili text normalization,
- Whisper Small ASR fine-tuning,
- QLoRA-based Swahili reasoning adaptation,
- independent ASR and reasoning evaluation,
- error diagnostics,
- and GPU inference benchmarking.

The measured ASR improvements are substantial, and the reasoning component improved from 0% to 97% Exact Match after changing the fine-tuning objective.

The decision not to use a monolithic multimodal model was deliberate. Given the T4's 14.56 GB VRAM, the available speech and reasoning datasets, the 48-hour assessment setting, and the need for reproducibility, a modular architecture provided a more defensible engineering solution.

The resulting design still provides multimodal functionality: **speech is transformed into language and then processed by a language reasoning model**. The architecture therefore prioritizes measurable performance, diagnostic transparency, resource efficiency, and production maintainability over unnecessary architectural complexity.

---

## Appendix A — Final Metrics

```text
ASR
Baseline WER:       1.4804
Fine-tuned WER:     0.4298
WER improvement:    70.97%

Baseline CER:       0.7637
Fine-tuned CER:     0.1224
CER improvement:    83.97%

Reasoning
Baseline EM:        0.00
QLoRA V1 EM:        0.00
QLoRA V2 EM:        0.97
Correct:            97/100
Errors:             3/100

Edge ASR
Parameters:         241,734,912
Average latency:    1.113 sec/sample
Peak GPU memory:    3.14 GB
GPU:                Tesla T4
```

## Appendix B — Model Components

| Component | Model | Adaptation |
|---|---|---|
| ASR | `openai/whisper-small` | Supervised fine-tuning |
| Reasoning | `Qwen/Qwen2.5-1.5B-Instruct` | QLoRA |
| Quantization | BitsAndBytes 4-bit NF4 | Reasoning inference/training |
| Dataset | `google/fleurs`, `sw_ke` | ASR |
| Reasoning data | Controlled synthetic Swahili benchmark | Reasoning |


## 11.1 Quantization strategy and local packaging

Task 4 is addressed with a component-specific optimization strategy rather than applying one compression method blindly to both models.

### Reasoning model

The Qwen2.5-1.5B-Instruct reasoning component was trained and evaluated with **4-bit NF4 quantization**, double quantization, and FP16 compute through BitsAndBytes. QLoRA updates only low-rank adapter parameters, allowing the adapted model to be stored as a small adapter rather than a second full-precision copy of the base model.

The reproducible adapter artifact is:

```text
models/qwen2.5-1.5b-qlora-v2/checkpoint-100/
```

### ASR model

Whisper Small was retained in FP16 for the reported deployment benchmark. This is a deliberate accuracy/resource trade-off: the model has 241,734,912 parameters, approximately 0.45 GB of FP16 weights, and measured peak GPU allocation of only 3.14 GB on the 14.56 GB T4. Because no separate quantized-ASR WER/CER experiment was completed, the report does not invent a quantized accuracy or latency gain.

### Local-serving package

The repository includes `src/deployment/serve.py` for checkpoint-based local pipeline configuration and `src/deployment/benchmark.py` for reproducible latency/throughput/memory measurements. Large model weights remain outside Git and are referenced through checkpoint paths.

### Optimization decision matrix

| Component | Strategy | Why | Status |
|---|---|---|---|
| Whisper Small | FP16 | Low measured memory footprint; protects measured ASR quality | Measured |
| Qwen2.5-1.5B | 4-bit NF4 + double quantization | Reduces base-model memory and enables QLoRA on T4 | Used |
| LoRA adapter | Adapter-only packaging | Avoids shipping a second full model copy | Used |
| Future ASR edge build | INT8/INT4 or ONNX/TensorRT | Potential latency/memory reduction | Requires target-device benchmark |

This makes the deployment claim precise: the project demonstrates **quantized reasoning inference and measured FP16 ASR efficiency**, while leaving hardware-specific ASR compression as a validated next experiment rather than an unsupported claim.
