> **Historical document: record of the initial 10-case version.** For the current 474-case hard benchmark and the probability calibration evaluation procedure, the root [README](../../README.md) and [BENCHMARK_PROTOCOL](../BENCHMARK_PROTOCOL.md) take precedence. The default case counts and test results of the old version described here do not represent the results of the currently distributed version.

# Final Review and Handover Status

> Historical document from the initial offline audit. For current results see [README](../../README.md) and [results/SUMMARY.md](../../results/SUMMARY.md).

**Verification date: 2026-11-09 (JST)**

## Conclusion

- The deliverable is **a repository whose offline implementation, data, and scoring code have been verified, for trying out in a real environment**.
- **Live requests to the four real model paths have not been run.** End-to-end operation using a GPU or an OpenAI API key, and measured accuracy, speed, or billed cost, are not guaranteed.
- No connection to or contract with the Jev API is required.

## Four Paths Compared

| Path | Model | Points to check against official specs |
|---|---|---|
| `luna-llm` | GPT-6 Luna | OpenAI `POST /v1/responses` + strict JSON Schema + `reasoning.effort=none` |
| `openai-decisions` | GPT-6 Luna (the ID officially specified for the Decisions API) | OpenAI `POST /v1/decisions` + array of 3 `choice` questions |
| `clef-flash` | Cloudflare/clef-flash | Dedicated SGLang `POST /v1/systemone` per the Cloudflare public model card |
| `d1-2b` | LiquidAI/d1-2B-GGUF | llama.cpp `POST /v1/systemone` per the Liquid AI model card; `model` omitted by default |

**A matching API model ID does not mean that the internal model weights or inference algorithm match.**

## Fixes Made During Review

1. Fixed an issue where the `model` field was required in the d1-2B request. It now follows the official llama.cpp example: it is omitted by default and sent only when `--d1-model` is specified.
2. Re-classified the `group` labels of the 10 hard cases to match the actual text. **The gold labels and input texts match the 48 source cases exactly**, as guaranteed by a data regeneration test.
3. Incomplete OpenAI Responses results, unexpected SystemOne answer types, and non-numeric or duplicate probability candidates are now treated as errors.
4. Fixed an aggregation issue in which Macro-F1 became `0.0` when all models failed. Unmeasured values are now `null`.
5. Added `case_comparison.csv`, which shows per-case **gold answers, model predictions, and match/mismatch** in table form.
6. To prevent arbitrary text from CSV being executed as a formula, strings whose leading character looks like a spreadsheet formula are now escaped.
7. Corrected README descriptions that did not match the 10-case version. Added an offline CI workflow in GitHub Actions that does not call any real API or model.

## Verification Performed in This Environment

```bash
python -m unittest discover -s tests -v
python -m scripts.http_contract_smoke
python -m scripts.systemone_pipeline_smoke
python -m scripts.verify_dataset
python benchmark.py --dry-run --limit 2
```

- **24 unit tests passed.**
- **Mock HTTP communication for the 4 paths succeeded** (note: no real API was called).
- **CLI integration test using two mock SystemOne servers succeeded**; summary, prediction, and per-case CSV output were confirmed.
- **The 48 synthetic source cases were regenerated with matching SHA-256 hashes**; the curated 10 cases also match the regenerated output exactly.
- Against a local server that cannot be reached, the run **exits with code 2 and explicitly reports an incomplete experiment and the number of cases not run**.
- **Re-verified after extracting the ZIP into a separate directory** (at distribution time).

## Unverified Items That Need Attention

- Live OpenAI Responses / Decisions API access, model usage permissions, billing, and full compatibility with actual response schemas.
- Actual GPU startup for Clef-Flash, SGLang compatibility, and GPU VRAM requirements.
- Loading the actual Liquid d1-2B GGUF model, the llama.cpp version, and the response schema.
- **The 10 cases are fictitious teaching material, and the gold labels have not been independently reviewed by power equipment experts.** Statistical significance, product superiority, and real-world operational accuracy cannot be concluded.
- The p95 and ECE values for 10 cases are reference indicators only. Do not conclude that calibration is good from a small number of successful samples.
- Latency for all paths includes API communication, hardware, and server configuration, so it is not a strict comparison of computation time for the same model.

## Primary Sources

- OpenAI Decisions API specification: https://developers.openai.com/api/reference/resources/decisions/methods/create
- OpenAI Decisions developer guide and pricing: https://developers.openai.com/api/docs/guides/decisions
- OpenAI GPT-6 Luna model and Responses: https://developers.openai.com/api/docs/models/gpt-6-luna
- Cloudflare Clef-Flash model card: https://huggingface.co/Cloudflare/clef-flash
- Liquid AI d1-2B GGUF model card: https://huggingface.co/LiquidAI/d1-2B-GGUF
