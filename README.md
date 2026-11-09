# Power Decision Benchmark

A reproducible benchmark for **structured labelling of power-equipment inspection records**, comparing generative LLMs, OpenAI's Decisions API, and local decision models. It measures exact-match accuracy, probability calibration, selective prediction, cause-contrast robustness, and latency on a hard set of 474 **synthetic** records.

> **Status: LIVE-VERIFIED on synthetic data (single environment).** All nine systems were run on the calibration (124) and held-out test (350) splits. Aggregated results are in [`results/SUMMARY.md`](results/SUMMARY.md). The numbers describe this benchmark on one V100 32 GB machine. They do not establish field performance.

Short paper (3 pages): [`docs/paper/main.pdf`](docs/paper/main.pdf).

## What is measured

1. **Accuracy.** Exact match over the three fields: `equipment` (6 options), `observation` (8), and `cause_status` (3: confirmed / suspected / unknown). Per-field accuracy is also reported, along with how often a suspected cause is wrongly labelled as confirmed.
2. **Probability quality.** Per-field Expected Calibration Error (ECE, 10 bins), multiclass Brier score, and reliability curves. Fields are scored separately because the option counts differ.
3. **Selective prediction.** A record is accepted when its lowest selected-option probability reaches a threshold. The threshold is chosen **only on the 124 calibration records** (lowest value with observed error ≤ 5% and at least 10 accepted cases) and then applied unchanged to the **350 test records**. Coverage, selective risk, and risk–coverage curves are reported.
4. **Robustness.** Corrections, retractions, negations, resolved events, cross-device confusion, and **cause-contrast triplets** (records that differ only in cause status).
5. **Systems.** Latency (p50/p95 per request), input/output tokens, and failures. Failed requests count against accuracy.

Confidence intervals come from a **cluster bootstrap over scenario families** (1000 resamples), because records from the same family are not independent.

**Probability sources.** The `confidence` field returned by some APIs is a concentration statistic, not the probability of the chosen option, so it is stored but never used for calibration. Native decision models return a distribution over the options directly. For generative models we use **output-token log-probabilities** (`--llm-logprobs`) and take the probabilities of candidate values at the first token of each field, renormalised over the candidates (`probability_source = logprobs_first_token_approx`). This is an approximation for multi-token values. Self-reported probabilities are not used.

## Results (test split, 350 records)

| System | Deployment | Exact (95% CI) | ECE equipment | Median latency |
|---|---|---|---|---|
| GPT-6 Luna (Responses, logprobs) | OpenAI API | 0.940 (0.861–0.991) | 0.006 | 1,309 ms |
| GPT-6 Luna (Decisions API) | OpenAI API | 0.920 (0.832–0.986) | 0.011 | 301 ms |
| Qwen3.8-27B (UD-Q4_K_M, logprobs) | V100 32 GB | 0.931 (0.851–0.984) | 0.011 | 1,998 ms |
| kev 9B (Q8_0, Ollama) | V100 32 GB | 0.880 (0.800–0.941) | 0.073 | 503 ms |
| kev 9B (BF16, Ollama) | V100 32 GB | 0.877 (0.797–0.937) | 0.072 | 1,263 ms |
| kev 0.8B (Q8_0, Ollama) | V100 32 GB | 0.554 (0.450–0.649) | 0.318 | 139 ms |
| kev 0.8B (BF16, Ollama) | V100 32 GB | 0.549 (0.447–0.641) | 0.317 | 221 ms |
| Clef-Flash (Q8_0, llama.cpp) | V100 32 GB | **0.954** (0.912–0.986) | 0.159 | 286 ms |
| d1-2B (Q8_0, llama.cpp) | V100 32 GB | 0.877 (0.791–0.948) | 0.072 | 116 ms |

Full tables (calibration, selective prediction, cause-contrast triplets, token counts, and the calibration split) are in [`results/SUMMARY.md`](results/SUMMARY.md).

- The top systems have overlapping confidence intervals, so the ranking is not established.
- Clef-Flash has the highest accuracy but the worst equipment calibration among the top systems.
- Among the systems that output probabilities, calibration varies by field. GPT-6 Luna (logprobs) is best for equipment (ECE 0.006). Qwen3.8-27B is best for observations (ECE 0.023). Both generative results use the first-token approximation.
- Latency comes from one run per system. OpenAI timings include network time, and local systems ran on one V100 32 GB. These are not a like-for-like speed comparison.

## Systems compared

| Backend | Model / route | Probabilities |
|---|---|---|
| `luna-llm` | GPT-6 Luna, OpenAI `POST /v1/responses`, JSON Schema | Output-token logprobs (first-token approximation) |
| `openai-decisions` | OpenAI `POST /v1/decisions`, model ID `gpt-6-luna` | Native, returned by the API |
| `qwen3.8-27b` | `unsloth/Qwen3.8-27B-GGUF` (UD-Q4_K_M), llama.cpp `/v1/chat/completions` | Logprobs (first-token approximation); thinking disabled; temperature 0 |
| `kev-9b`, `kev-9b-bf16` | `kev:9b-q8_0`, `kev:9b-bf16`, Ollama `/v1/systemone` | Native |
| `kev-0.8b`, `kev-0.8b-bf16` | `kev:0.8b-q8_0`, `kev:0.8b-bf16`, Ollama `/v1/systemone` | Native |
| `clef-flash` | `Cloudflare/clef-flash` (Q8_0 via `ggml-org/Clef-Flash-GGUF`), llama.cpp `/v1/systemone` | Native |
| `d1-2b` | `LiquidAI/d1-2B-GGUF` (Q8_0), llama.cpp `/v1/systemone` | Native |

Caveats: the same model ID on the Responses and Decisions APIs does not imply identical internals. Clef-Flash, d1-2B, kev, and Qwen differ in quantisation, runtime, and hardware path.

## Data

| File | Purpose | Records |
|---|---|---:|
| `data/hard_inspection_cases.jsonl` | Full hard set (default) | 474 |
| `data/hard_calibration.jsonl` | Threshold selection | 124 |
| `data/hard_test.jsonl` | Held-out evaluation | 350 |
| `data/hard_split_manifest.json` | Seed, case IDs, family IDs, SHA-256 of each file | — |
| `data/hard_gold_audit.csv` | Gold labels with short justifications | 474 |
| `data/legacy/demo_10_cases.jsonl` | Ten cases for demonstrations | 10 |
| `data/legacy/synthetic_inspection_cases.jsonl` | Earlier 48-case set (kept for compatibility) | 48 |

Generated by `scripts/make_hard_dataset.py`; the split is made by `scripts/make_calibration_split.py`. The 474 records consist of 432 template-style variants and 42 hand-written cases, grouped into 47 scenario families. The labels were written by the authors, **not by field engineers**, and the data must not be used for operating decisions or safety work.

## Quick start (offline checks, no keys or GPUs)

Python 3.10+ with the standard library only.

```bash
python -m unittest discover -s tests -v
python -m scripts.verify_dataset
python -m scripts.http_contract_smoke
python -m scripts.systemone_pipeline_smoke
python -m scripts.probability_smoke
python benchmark.py --dataset data/hard_test.jsonl --limit 2 --llm-logprobs --dry-run
```

## Running the live benchmark

Set `OPENAI_API_KEY` in your shell (never commit it). Start the local servers first:

```bash
# Clef-Flash (SGLang, from the model card; reference settings for a single H200)
docker run --gpus all --shm-size 32g -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface --ipc=host \
  lmsysorg/sglang:dev-clef sglang serve --model-path Cloudflare/clef-flash --host 0.0.0.0 --port 30000

# Clef-Flash GGUF via llama.cpp (what the published results used, Q8_0)
llama-server -m Clef-Flash-Q8_0.gguf --host 127.0.0.1 --port 8081 -b 8192 -ub 8192

# d1-2B via llama.cpp (llama.cpp b11960 / v0.8.0 or later)
llama-server -hf LiquidAI/d1-2B-GGUF:Q8_0 --host 127.0.0.1 --port 8080 -b 8192 -ub 8192

# Qwen3.8-27B via llama.cpp (chat completions with logprobs)
llama-server -m Qwen3.8-27B-UD-Q4_K_M.gguf --host 127.0.0.1 --port 8082 -c 8192 -b 8192 -ub 2048 --jinja

# kev via Ollama (requires a recent Ollama; the tags are listed at https://ollama.com/library/kev)
ollama pull kev:9b-q8_0
```

Run the calibration and test splits with the same settings:

```bash
python benchmark.py --dataset data/hard_calibration.jsonl \
  --backends luna-llm,openai-decisions,qwen3.8-27b,kev-9b,kev-9b-bf16,kev-0.8b,kev-0.8b-bf16,clef-flash,d1-2b \
  --llm-logprobs --clef-url http://127.0.0.1:8081 --d1-url http://127.0.0.1:8080 \
  --qwen-url http://127.0.0.1:8082 --kev-url http://127.0.0.1:11434 \
  --outdir results/raw/calibration

python benchmark.py --dataset data/hard_test.jsonl \
  --backends luna-llm,openai-decisions,qwen3.8-27b,kev-9b,kev-9b-bf16,kev-0.8b,kev-0.8b-bf16,clef-flash,d1-2b \
  --llm-logprobs --clef-url http://127.0.0.1:8081 --d1-url http://127.0.0.1:8080 \
  --qwen-url http://127.0.0.1:8082 --kev-url http://127.0.0.1:11434 \
  --outdir results/raw/test
```

Benchmark runs stop with exit code 2 if any requests fail, and they report how many were not attempted. A complete run is required for a valid comparison. Use `--allow-partial` only for debugging.

Then run the audit, the paired comparison, the charts, and the summary:

```bash
python -m scripts.probability_audit \
  --calibration results/raw/calibration/predictions.jsonl \
  --test results/raw/test/predictions.jsonl \
  --manifest data/hard_split_manifest.json \
  --risk-budget 0.05 --bootstrap 1000 \
  --outdir results/probability_audit
mv results/probability_audit/case_probability.csv results/raw/probability_audit/
python -m scripts.paired_comparison results/raw/test/predictions.jsonl --out results/probability_audit/paired_test.csv
python -m scripts.plot_probability_audit results/probability_audit
python -m scripts.make_summary
```

Raw per-case predictions (`results/raw/`) are not published. The published files are the summary tables, the audit report, the charts, and the paired comparison.

## Repository layout

```
benchmark.py     runner for all systems (same prompt and labels for every system)
data/            474 hard cases, calibration/test splits, split manifest, gold audit
data/legacy/     earlier 48-case set and its 10-case demonstration subset
scripts/         dataset generators, probability audit, summary and paper builders, smoke tests
tests/           offline unit tests
results/         published summaries (SUMMARY.md, probability_audit/); raw predictions stay local
docs/            evaluation protocol and probability notes; paper/ (LaTeX and PDF); history/ (earlier audits)
```

## Paper

A short write-up is in [`docs/paper/`](docs/paper/main.pdf). It is generated from the published tables and charts:

```bash
python3 scripts/make_paper.py
cd docs/paper && pdflatex main.tex && pdflatex main.tex
```

## Quality notes

- Brier and NLL are computed only on valid probability distributions, and the number of invalid cases is reported. ECE (top-label) uses cases where the predicted label matches the most probable option. Mismatches are counted separately.
- Accuracy and coverage use **all planned cases as the denominator**. A case without a usable probability is never auto-accepted.
- Stored `confidence` values are not used for ECE, Brier, or NLL.
- A zero probability on the gold label is floored at `1e-12` in the NLL, and the count is reported.
- Synthetic writing style and templates differ from real maintenance records. Do not read the accuracy or calibration as field performance.
- Before sending real inspection data to any external API, check the contract, information-security requirements, anonymisation, and third-party disclosure rules.
- Record the run date, Python and OS versions, exact model IDs and weight versions, quantisation, server versions, GPU, and network conditions. Latency figures without these are only observations.

More detail: [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md), [docs/PROBABILITY_REVIEW.md](docs/PROBABILITY_REVIEW.md), and the other documents in `docs/`.

## License

- **Code and documentation** in this repository: Apache License 2.0 ([LICENSE](LICENSE)).
- **Synthetic data** (`data/`, including `data/legacy/`): fictional records. They do not refer to real equipment, companies, or people.
- **Models** are not included in this repository. Obtain them from the providers.

| Model | License | Source |
|---|---|---|
| Clef-Flash | Apache-2.0 (base: Qwen/Qwen3.5-9B) | https://huggingface.co/Cloudflare/clef-flash |
| d1-2B (GGUF) | lfm1.0 (Liquid AI custom license) | https://huggingface.co/LiquidAI/d1-2B-GGUF |
| Qwen3.8-27B (GGUF, Unsloth) | Apache-2.0 | https://huggingface.co/unsloth/Qwen3.8-27B-GGUF |
| kev (Ollama) | Apache-2.0 | https://ollama.com/library/kev |
| GPT-6 Luna (OpenAI API) | Weights not public. Use is governed by OpenAI's terms | https://developers.openai.com/api/docs/models/gpt-6-luna |
