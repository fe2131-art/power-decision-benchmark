# Evaluation Protocol: Independent Power Equipment Evaluation Based on the TypeSafe/Jev Benchmark Designs

> Historical document from the initial offline audit. For current results see [README](../README.md) and [results/SUMMARY.md](../results/SUMMARY.md).

## Primary Sources Consulted (verified 2026-11-09)

| Source | Content and application to this evaluation |
|---|---|
| [TypeSafe WorkflowEvals (official GitHub)](https://github.com/typesafe-ai/WorkflowEvals) / [experiment site](https://evals.typesafe.ai/) | Decomposes realistic business workflows into questions plus code, and measures final-action agreement, cost, and time. The reference for correct answers uses the agreement distribution among large models. This repository uses **designed labels on fictional records** and does not pretend to report the same "agreement rate." |
| [TypeSafe Confidence (official)](https://docs.typesafe.ai/confidence) | A Choice's `confidence` is `(p_max−1/n)/(1−1/n)`. It is not the probability of being correct, nor new evidence. Calibration and thresholds use the candidate probabilities, and the provider's confidence is stored as a separate column. |
| [Evaluating and Benchmarking the System One Model Jev (research code)](https://github.com/AppliedMachineLearning-Lab/jev-benchmarking) / [paper](https://arxiv.org/abs/2609.37647) | Covers 37 datasets, probability calibration of Choices, selective prediction, per-language differences, and threshold tuning. Freezing experimental conditions and separating metrics serve as references. |
| [jev-benchmarks (independent evaluation)](https://github.com/AbdelStark/jev-benchmarks) / [PROTOCOL](https://github.com/AbdelStark/jev-benchmarks/blob/main/docs/PROTOCOL.md) | Accuracy, Macro-F1, multiclass Brier, NLL, top-label ECE, coverage at an acceptable empirical error rate, failures, latency, and a series-aware bootstrap. These are implemented and applied here. |
| [Sys1Cal-v1](https://arxiv.org/abs/2609.35342) | Artificial problems whose correct probability distribution is known in advance and calibration to observed correctness are **different questions**. This dataset has only definitive gold labels and no supervision values for the true probability distribution. |

## 0. Notes on Comparison

**No Jev API or TypeSafe weights are run at all.** The models are GPT-6 Luna via the OpenAI Responses API, the OpenAI Decisions API, local Clef-Flash, and local d1-2B. Performance is measured in the user's own execution environment; the providers' reported figures are not substituted into this data.

Each pipeline receives the same records and label descriptions, but API-specific prompt formatting, token counting, and model architectures are not identical. Do not confuse a generative LLM's self-reported probabilities with native probabilities.

## 1. Data and Correct Answers

All 474 items are fictional equipment maintenance records composed by the author (408 are variations of equipment × event templates across 34 series, 24 are normal-state confirmations across 6 series, and 42 are individually written across 7 series). There are three kinds of correct answers: the equipment, the current anomaly, and the **evidence level of the description of the anomaly's cause**. The cause label does not diagnose actual physical causality; it labels **whether the document states the cause as presumed or confirmed**. A separate review by equipment experts has not been conducted.

The hard-question set includes time series, cancellations, confusion with other equipment, and retraction of a confirmed cause. `family_id` is the unit that groups similar texts, such as groups of 12, 6, or 4 items. **474 rows ≠ 474 independent trials.**

## 2. Holdout

Before looking at any model output, the data is split by family and stratified by difficulty, using `seed=20261109` and `fraction=0.30`. Tuning: 124 items (14 series). Test: 350 items (33 series). Fixed IDs, series, and hashes are recorded in `data/hard_split_manifest.json`.

Threshold changes, model selection, and rubric edits are based only on the tuning data. If optimization is performed after looking at the final test set, a new test split is required.

## 3. Metrics

| Metric | Definition |
|---|---|
| Exact accuracy | `n_3-field matches / n_all scheduled items`. Failures are not treated as correct. |
| Field accuracy | For each of the three fields, the denominator is all scheduled items, including failures. |
| Brier (per field) | `mean_i Σ_k (p_ik−1[k=y_i])²`; lower is better. Evaluated per field because the value range and baseline differ with 6/8/3 candidates. |
| NLL (per field) | `mean_i −log(max(p_i,correct,1e−12))`. A lower bound is applied for numerical computation, and the number of items with zero probability on the correct answer is reported. |
| ECE (top-label, 10 bins) | Weighted average of the per-bin difference between the predicted candidate's probability and the actual frequency of correct answers. Computed only when the distribution is valid and the selection matches the maximum probability. **Bins with few samples are unstable.** |
| `provider_confidence` | Each API's confidence is **not included in Brier/ECE**; the mean and distribution of the provider-supplied values and the selected probabilities are reported separately. |
| `probability_case_coverage` | Number of cases in which the probability distributions for all three fields are complete, valid, and consistent with the selection / all scheduled items. |
| `Coverage(τ)` | Number of records in which the selected probability is at least `τ` for all fields / all scheduled items. Invalid probabilities and API failures are not counted. |
| `Selective Risk(τ)` | Among accepted records, the proportion in which any of the three fields is incorrect. If zero records are accepted, the value is **undefined** (it is not shown as zero). |
| Risk–Coverage curve / AURC | Operating points at the specified set of thresholds, and the area under the curve for cases with complete probabilities. The valid-probability rate is reported alongside AURC. |
| Bootstrap 95% CI | Resamples test series (`family_id`). This is a **descriptive interval for synthetic series**, not a guarantee of extrapolation to a real-world population. |
| Cause contrast triplets | Proportion of triplets in which all three items are answered correctly. Each triplet uses the same equipment × event × template style and changes **only the cause state**. Measures sensitivity to causal-expression differences arising from templates. |
| Success / failure / latency / cost | Completed count, failure count, p50 and p95, whether usage was retrieved, and an OpenAI price estimate (null if not retrieved). |

## 4. Freezing the Threshold

On the **tuning** set, among candidates satisfying "empirical error rate `≤0.05`, accepted count `≥10`", select the value that yields the largest number of automatic acceptances. Candidates are the `min(p_selected)` values observed in the tuning examples. If no candidate qualifies, the result is "threshold selection not possible," and the **test set has zero automatic acceptances**. Test predictions are not used for threshold selection. This is an empirical risk constraint and **not a guarantee of a 5% error rate in the future**.

Thresholds may be set separately for each model (coverage comparison under the same quality constraint). Comparisons at fixed values are also output to `risk_curve.csv`. Failures remain in the denominator as all scheduled items.

## 5. Calibration Pitfalls

- Native candidate probabilities: outputs from the SystemOne family, namely OpenAI Decisions, Clef, and d1.
- **Verbalized LLM probability**: numbers that GPT-6 Luna was asked to output for itself via `--llm-probabilities`. The API does not promise direct candidate probabilities. Whether self-reported probabilities are calibrated is tested as an experiment, but these are not model outputs produced by the same mechanism as native probabilities.
- In Jev, `confidence` is a measure of how peaked the distribution is (its meaning changes with the number of candidates). Do not confuse it with choice probabilities, and do not set thresholds on a single "confidence value" shared across all models.
- Calibration depends on the task, language, model version, question wording, and data distribution. The results of the Japanese synthetic spot-check memo do not demonstrate quality in field operations.
- Explicitly report: the clamping of zero in NLL; the rounding tolerance for probability sums up to ±2% and the renormalization; the number of missing probabilities; and the number of predictions that disagree with argmax.
- **Robustness to option-order changes and run-to-run variation from repeated execution** are separate experiments. The current script uses a frozen fixed order and a single run, and these metrics are **not implemented**.

## 6. Execution and Reproduction

Use the 124-item tuning set and the 350-item test set from the root README, and run `python -m scripts.probability_audit`. The metrics in the tables can be recomputed from `predictions.jsonl` and the fixed manifest.

## 7. Limitations

- **Performance values have not been obtained.** The simulated probabilities on the test set do not represent the performance of real models.
- Template similarity and author bias have not been completely eliminated.
- Because inference methods, tokenization, and hardware differ, comparisons of "pure model performance" and "response time of a practical system" should be interpreted separately.
- Adoption for safe operation requires real data whose correct answers have been verified by field staff, along with a formal risk standard.
