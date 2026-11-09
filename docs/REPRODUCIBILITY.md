> **Historical record: the earlier 10-case version.** For the current 474 hard cases and the probability calibration evaluation procedure, see the root [README](../README.md) and [BENCHMARK_PROTOCOL](BENCHMARK_PROTOCOL.md). The default case counts and test results of the earlier version described here do not represent the results of the currently distributed version.

# Live Verification and Reproducibility Checklist

> Historical document from the initial offline audit. For current results see [README](../README.md) and [results/SUMMARY.md](../results/SUMMARY.md).

This repository is the **entry point for evaluation**. The current code has not yet verified backend real inference.

## 0. Security and Use

- **Do not send real customer, equipment, or employee data to external APIs.** The bundled data is limited to the default 10 synthetic cases and the original 48 synthetic cases.
- Limit use to generating analysis labels for inspection records. Do not use it for power outage decisions, operational control, or safety decisions.
- Set API keys as environment variables. Do not commit `.env`, API responses, internal data, or result files to Git.

## 1. Initial Offline Verification

```bash
python --version
python -m unittest discover -s tests -v
python -m scripts.http_contract_smoke
python -m scripts.systemone_pipeline_smoke
python -m scripts.verify_dataset
python -m scripts.check_environment
python benchmark.py --dry-run --limit 4
```

`--dry-run` makes no API calls and prints request samples to standard output.

## 2. Live Verification of the Two OpenAI Paths Only

After setting the OpenAI key as an environment variable:

```bash
python benchmark.py --backends luna-llm,openai-decisions --limit 4 --outdir results/openai-smoke
```

Be sure to **inspect `results/openai-smoke/predictions.jsonl`** and confirm that there are no HTTP 4xx errors, parse failures, missing Usage, or missing fields.
**`gpt-6-luna` is the default model ID in the current code. `--llm-model` and `--decisions-model` can be specified independently. Service availability and model support must be verified against your contract and the latest documentation.**

## 3. Local Decision Models

- d1-2B GGUF: Start it on a compatible llama.cpp `/v1/systemone`, and specify `--d1-url`. The model field is omitted by default; specify `--d1-model` only if the server requires it.
- Clef-Flash: Start it on a compatible SGLang `/v1/systemone`, and specify `--clef-url` (and `--clef-model` if needed).
- **This is not a local LLM generative chat API.** A model-specific decision API is required.

After loading a model, run a small check first:

```bash
python benchmark.py --backends d1-2b --limit 4 --outdir results/d1-smoke
python benchmark.py --backends clef-flash --limit 4 --outdir results/clef-smoke
```

Always confirm that `complete_run=true` in `summary.json` and that `error=null` for all cases in `predictions.jsonl`.

## 4. Comparison of All Four Methods

```bash
python benchmark.py \
  --backends luna-llm,openai-decisions,d1-2b,clef-flash \
  --threshold 0.85 --outdir results/demo10
```

**Items to fix**: input JSONL and its SHA256, model IDs and revisions, quantization, server commit/version, GPU/VRAM/CUDA, OS and Python, inference parallelism, warm-up count, number of repetitions, API contract conditions, date and time, external model rate limits and caching, thresholds, and the evaluation set.

### Outputs

- `summary.csv` / `summary.json`: metrics per backend
- `case_comparison.csv`: a list of correct answers, predictions, match/mismatch status, and errors for each of the 10 cases × models
- `predictions.jsonl`: sequential predictions, normalized model responses, latency, Usage, and errors (**review the contents before sharing externally**)
- `docs/AUDIT_REPORT.md`: verification status and limitations at the mock stage

**Required checks**:

1. The planned, attempted, and successful counts must match. Do not use accuracy that ignores failures or skips as the primary comparison.
2. Report both overall accuracy and per-item accuracy. Check the number of cases where a cause estimate is mistakenly treated as confirmed.
3. Do not force-fill ECE or Brier for LLMs, since they have no candidate distribution.
4. Speed comparisons are affected by network and GPU differences. They are not measurements of model compute time alone.
5. Do not claim performance on these same 10 cases after adjusting thresholds while looking at them. Re-evaluate on external, independent data.

## 5. If Errors Occur

- `OPENAI_API_KEY is not set`: Set the environment variable.
- `HTTP 400/422`: Check the API schema, model ID, and reasoning effort against the latest specification. Commit the fix and record it.
- `Connection failure` / `connection refused`: Check the server, port, and firewall. Confirm TCP listening with `python -m scripts.check_environment`.
- `complete_run=false`: The experiment failed partway through. Check the error column in `results/.../predictions.jsonl` and `summary.json`.
- Out of memory: Record VRAM, quantization, context length, and serving method, and rerun as a separate configuration.

If the actual server's request/response shapes differ, update the adapters in `benchmark.py` and the test fixtures, and do not mix results from before and after the update.

These 10 cases are for presentation demos. Simple metric comparisons can be displayed, but do not make statistical judgments about which method performs better.
