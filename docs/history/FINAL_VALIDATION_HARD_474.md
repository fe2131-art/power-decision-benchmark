# 474-Case Probability Calibration Edition: Final Validation (2026-11-09)

> Historical document from the initial offline audit. For current results see [README](../../README.md) and [results/SUMMARY.md](../../results/SUMMARY.md).

## Additions

- Reproducible synthetic dataset of 474 hard cases. Breakdown: 408 equipment × event template variants, 24 normal-check cases, and 42 individually written boundary cases. Classified into 47 series.
- Automatic acceptance threshold selected on 124 cases (14 series) and evaluated on a non-overlapping hold-out set of 350 cases (33 series).
- Jev model execution is not included. OpenAI Responses / Decisions, Clef-Flash, and d1-2B are applied to the same equipment records as 3-way choice items.
- Brier, NLL, Top-label ECE, Risk–Coverage, AURC, series-level bootstrap 95% CI, and control triplets, computed from the choice probability distribution.
- LLM self-reported probabilities and native Decisions API probabilities are distinguished. `provider_confidence` is not used as the correct-answer probability.
- Official SGLang / llama.cpp serving examples for users to try in their own environment, and SVG figures generated with the standard library only.

## Verification Performed

- `python -m unittest discover -s tests -q` — 44 tests passed.
- `python -m scripts.verify_dataset` — regeneration of the old 48, the 10, and the new 474 cases; Gold labels and splits match exactly.
- `python -m scripts.http_contract_smoke` — simulated API send/receive succeeded for the 4 paths.
- `python -m scripts.systemone_pipeline_smoke` — simulated SystemOne send/receive and aggregation succeeded for the local decision model.
- `python -m scripts.probability_smoke` — aggregation succeeded with mock probabilities for 4 paths × 124 + 350 cases.
- `python benchmark.py --dry-run --limit 1 --llm-probabilities` — request creation confirmed with defaults for the full dataset.
- After the ZIP was created, it was extracted to a separate directory and the above tests and the SHA256SUMS check were re-run (verified at build time).

## Explicitly Unverified

The OpenAI live API and actual weight inference for Clef-Flash / d1-2B were not run. Actual latency, accuracy, and probability calibration values were not obtained in this environment. An external expert audit of the Gold labels on field maintenance data has also not been performed. All evaluations are limited to fictitious data.

## Primary Sources

- https://docs.typesafe.ai/confidence
- https://github.com/typesafe-ai/WorkflowEvals
- https://github.com/AbdelStark/jev-benchmarks
- https://github.com/AppliedMachineLearning-Lab/jev-benchmarking
- https://huggingface.co/Cloudflare/clef-flash
- https://huggingface.co/LiquidAI/d1-2B-GGUF
- https://developers.openai.com/api/docs/guides/decisions
