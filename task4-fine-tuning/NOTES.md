# Task 4 Technical Notes — Cross-Task Observations

Practical notes and lessons learned from fine-tuning across 4 industrial vision tasks.

---

## 1. Prompt Quality Is Critical

The single biggest factor affecting accuracy was prompt quality — specifically, using the SAME detailed class definitions in training that the frontier model used during benchmarking.

**Lesson learned on steel surface**: Initial training used shorter, less precise class definitions than the benchmarking prompt. Result: 48.6% (SEAL). After switching to the exact benchmarking definitions: 66.7%. Same training images, same hyperparameters — only the prompt changed.

The definitions must include:
- Specific visual features (not just class names)
- Distinguishing characteristics between similar classes
- Domain-specific terminology that grounds the classification

---

## 2. Line Stripping for CoT Examples

The user prompt for CoT training examples must NOT contain the "Respond with JSON" instruction. This is done by stripping the last 2 lines of the prompt:

```python
PROMPT_NO_JSON = '\n'.join(PROMPT.split('\n')[:-2])
```

**Why 2 lines, not 1**: The prompt ends with:
```
(empty line)
Respond with ONLY a JSON object:
{"defect_class": "<...>"}
```

Stripping only the last line removes the JSON template but leaves "Respond with ONLY a JSON object:" — which contradicts the CoT training where the assistant outputs a description.

**Lesson learned on steel surface**: Initial code used `lines[:-1]`. The CoT prompt still said "Respond with ONLY a JSON object" while the assistant response was a long description. This confused the model and SEAL performed worse than Direct (48.6% vs 54.4%).

---

## 3. Learning Rate Matters

All 4 tasks converged well with `LR = 2e-5`. Lower rates (1e-5, 5e-6) consistently underperformed.

**Lesson learned on steel surface**: Initial SEAL training used LR=1e-5 for 15 epochs, then LR=5e-6 for 25 more epochs. Result: 48.6%. Same data with LR=2e-5 for 40 epochs: 55.8% → 66.7% (after prompt fix).

The cosine schedule with 10% warmup handles the LR decay automatically — no need to manually reduce.

---

## 4. Training Images Per Class

| Task | Images/class | Total | SEAL Accuracy |
|------|-------------|-------|---------------|
| Granulometry | 2 | 18 | 79.6% |
| Steel Surface | 5 | 30 | 66.7% |
| UHCS Microstructure | 6 | 30 | 68.4% |
| Weld Defects | 6 | 24 | 75.8% |

Granulometry achieved the best results with the fewest images (2/class) because:
- The visual features are more distinct (particle size + gap patterns)
- The GSD provides a quantitative anchor for size estimation
- 9 classes but each has a clear visual signature

Steel surface needed 5/class because:
- 6 classes with subtle visual differences (inclusion vs scratches)
- 200×200 grayscale images have less information
- Initial run with 3/class: 55.8%. With 5/class: 66.7% (+11pp)

---

## 5. Class Imbalance and Rare Classes

UHCS had severe imbalance: spheroidite (372) vs pearlite+widmanstatten (5).

**Decision**: Dropped pearlite+widmanstatten (only 2 available in train pool). With 5 classes instead of 6, accuracy improved from 62.5% → 68.4%.

**Rule of thumb**: If a class has fewer than 3 training images available, consider dropping it or merging with a similar class.

---

## 6. Image Format in API Calls

- Steel surface (NEU-CLS): `.jpg` images → `data:image/jpeg;base64,...`
- UHCS micrographs: `.png` images → `data:image/png;base64,...`
- RIAWELC radiographs: `.png` images → `data:image/png;base64,...`

Using the wrong MIME type doesn't crash the API but may affect image processing quality.

---

## 7. SEAL Teacher Selection

GPT-4.1 was used as the SEAL teacher for all 4 tasks, even though GPT-5 scored higher on UHCS (80.0% vs 71.7%). Reasons:
- GPT-4.1 is 3-4× faster (2-3s vs 8-11s per image)
- GPT-4.1 supports temperature control (GPT-5 is locked at t=1)
- GPT-4.1 is cheaper
- For SEAL, the teacher doesn't need to be accurate — it just needs to describe visual features given the correct answer

---

## 8. Evaluation Temperature

All evaluations use `temperature=0.1` with `do_sample=True`. This gives near-deterministic outputs while avoiding the edge cases of `temperature=0` (which can cause repetition loops in some models).

**Lesson learned**: Early steel surface augmented eval used `temperature=0.7` (copy-paste from the GPT-4.1 generation code). This added unnecessary variance to evaluation results.

---

## 9. max_new_tokens for Evaluation

All evaluations use `max_new_tokens=256`, matching the granulometry setup. This is important for SEAL-trained models that may output a description before the JSON — shorter limits could truncate the JSON.

**Lesson learned**: Early steel surface direct eval used `max_new_tokens=80`. While sufficient for JSON-only responses, it was inconsistent with the augmented eval and could cause issues if the model outputs any preamble.

---

## 10. Confusion Patterns Across Tasks

Each task has characteristic confusion pairs:

| Task | Main Confusion | Why |
|------|---------------|-----|
| Granulometry | A8 ↔ B8 (coarse vs medium at 8mm) | Small particles, subtle gap differences |
| Steel Surface | inclusion → scratches (63%) | Both have dark elongated features |
| UHCS | spheroidite+widmanstatten → scattered | Requires detecting two co-existing features |
| Weld Defects | cracks → lack_of_penetration (45%) | Both are dark lines in radiographs |

The SEAL approach specifically helps with these confusions because the CoT descriptions include contrastive reasoning ("this is NOT scratches because the marks are irregularly shaped blobs, not sharp linear grooves").

---

## 11. JSONL Caching

All augmented notebooks cache the generated training data to `training_data_augmented.jsonl`. If the file exists, it's loaded instead of regenerating. This saves API costs and time when retraining with different hyperparameters.

**Important**: When changing the prompt or SEAL prompt, DELETE the cached JSONL file before re-running. Otherwise the old (wrong) training data will be reused.

---

## 12. Hardware Notes

All training was done on Azure ML `slm-workbench` with 2x V100 16GB.

Model loading: `max_memory={0: '6GiB', 1: '15GiB'}` — puts most of the model on GPU 1, leaving GPU 0 for activations and gradients.

Key calls:
- `base_model.enable_input_require_grads()` — required for gradient checkpointing with LoRA
- `model.gradient_checkpointing_enable()` — saves ~2-3 GB VRAM
- `use_fast=False` on processor — avoids the Qwen2VLImageProcessor warning
- `dtype=torch.bfloat16` (not `torch_dtype=`) — the newer transformers API
