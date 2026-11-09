# Results summary (test split, 350 cases)

Comparison on 474 synthetic power-equipment inspection records (calibration 124, test 350; split by scenario family). Raw per-case predictions (`results/raw/`) are not published. Values in parentheses for Exact are 95% cluster-bootstrap confidence intervals over scenario families. Probabilities are first-token approximations of output-token logprobs (luna-llm, Qwen) or native probabilities of the decision models (all others). Qwen was run with temperature=0; the other methods use each service's default sampling settings.

## Accuracy and probability calibration (test split, 350 cases)

| Method | Deployment | Exact (95% CI) | ECE equipment | ECE observation | ECE cause | Brier equipment | Brier observation | Brier cause | Coverage @ 5% risk budget | Error rate of accepted | Cause-contrast triplets |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-6 Luna (Responses, logprobs) | OpenAI API | 0.940 (0.861–0.991) | 0.006 | 0.045 | 0.008 | 0.011 | 0.093 | 0.018 | 0.977 | 0.056 | 0.990 |
| GPT-6 Luna (Decisions API) | OpenAI API | 0.920 (0.832–0.986) | 0.011 | 0.031 | 0.006 | 0.027 | 0.089 | 0.011 | 1.000 | 0.080 | 1.000 |
| Qwen3.8-27B (UD-Q4_K_M, llama.cpp, logprobs) | local V100 32GB | 0.931 (0.851–0.984) | 0.011 | 0.023 | 0.004 | 0.046 | 0.335 | 0.008 | 0.806 | 0.071 | 1.000 |
| kev 9B (Q8_0, Ollama) | local V100 32GB (Ollama) | 0.880 (0.800–0.941) | 0.073 | 0.062 | 0.066 | 0.033 | 0.098 | 0.057 | 0.997 | 0.120 | 0.942 |
| kev 9B (BF16, Ollama) | local V100 32GB (Ollama) | 0.877 (0.797–0.937) | 0.072 | 0.059 | 0.064 | 0.033 | 0.101 | 0.058 | 0.997 | 0.120 | 0.923 |
| kev 0.8B (Q8_0, Ollama) | local V100 32GB (Ollama) | 0.554 (0.450–0.649) | 0.318 | 0.356 | 0.319 | 0.351 | 0.536 | 0.356 | 0.069 | 0.083 | 0.683 |
| kev 0.8B (BF16, Ollama) | local V100 32GB (Ollama) | 0.549 (0.447–0.641) | 0.317 | 0.348 | 0.319 | 0.349 | 0.539 | 0.355 | 0.094 | 0.061 | 0.683 |
| Clef-Flash (Q8_0, llama.cpp) | local V100 32GB | 0.954 (0.912–0.986) | 0.159 | 0.137 | 0.054 | 0.076 | 0.073 | 0.024 | 0.991 | 0.040 | 1.000 |
| d1-2B (Q8_0, llama.cpp) | local V100 32GB | 0.877 (0.791–0.948) | 0.072 | 0.093 | 0.059 | 0.062 | 0.143 | 0.029 | 0.863 | 0.046 | 0.981 |

ECE and Brier are reported only for methods that output probabilities. Coverage and error rate use the threshold selected on the 124 calibration cases and applied unchanged to the test split. The 5% budget is an empirical evaluation condition, not a safety guarantee.

## Latency and tokens per case (test split)

| Method | Deployment | Cases | Failed | p50 (ms) | p95 (ms) | Mean (ms) | Mean input tokens | Mean output tokens |
|---|---|---|---|---|---|---|---|---|
| GPT-6 Luna (Responses, logprobs) | OpenAI API | 350 | 0 | 1,309 | 6,642 | 2,077 | 899 | 24.1 |
| GPT-6 Luna (Decisions API) | OpenAI API | 350 | 0 | 301 | 401 | 317 | 1056 | 0.0 |
| Qwen3.8-27B (UD-Q4_K_M, llama.cpp, logprobs) | local V100 32GB | 350 | 0 | 1,998 | 2,223 | 1,999 | 648 | 29.0 |
| kev 9B (Q8_0, Ollama) | local V100 32GB (Ollama) | 350 | 0 | 503 | 510 | 502 | 935 | 0.0 |
| kev 9B (BF16, Ollama) | local V100 32GB (Ollama) | 350 | 0 | 1,263 | 1,297 | 1,267 | 935 | 0.0 |
| kev 0.8B (Q8_0, Ollama) | local V100 32GB (Ollama) | 350 | 0 | 139 | 146 | 139 | 935 | 0.0 |
| kev 0.8B (BF16, Ollama) | local V100 32GB (Ollama) | 350 | 0 | 221 | 230 | 221 | 935 | 0.0 |
| Clef-Flash (Q8_0, llama.cpp) | local V100 32GB | 350 | 0 | 286 | 288 | 284 | 939 | 0.0 |
| d1-2B (Q8_0, llama.cpp) | local V100 32GB | 350 | 0 | 116 | 120 | 115 | 1025 | 0.0 |

Latency is wall-clock time measured per request. Each method was run on its own, one after another. OpenAI API timings include network latency; local timings were measured on a single V100 32GB. These numbers are not a like-for-like comparison of raw model speed.

## Calibration split (124 cases, used for threshold selection)

| Method | Exact | Cases | Failed |
|---|---|---|---|
| GPT-6 Luna (Responses, logprobs) | 0.984 | 124 | 0 |
| GPT-6 Luna (Decisions API) | 0.968 | 124 | 0 |
| Qwen3.8-27B (UD-Q4_K_M, llama.cpp, logprobs) | 0.927 | 124 | 0 |
| kev 9B (Q8_0, Ollama) | 0.960 | 124 | 0 |
| kev 9B (BF16, Ollama) | 0.952 | 124 | 0 |
| kev 0.8B (Q8_0, Ollama) | 0.661 | 124 | 0 |
| kev 0.8B (BF16, Ollama) | 0.653 | 124 | 0 |
| Clef-Flash (Q8_0, llama.cpp) | 0.968 | 124 | 0 |
| d1-2B (Q8_0, llama.cpp) | 0.919 | 124 | 0 |

## Reproduction

```bash
python -m scripts.probability_audit --calibration results/raw/calibration/predictions.jsonl --test results/raw/test/predictions.jsonl --manifest data/hard_split_manifest.json --outdir results/raw/probability_audit
python -m scripts.make_summary
```
