> **Historical document: record of the initial 10-case version.** For the current 474-case hard benchmark and the probability calibration evaluation procedure, the root [README](../../README.md) and [BENCHMARK_PROTOCOL](../BENCHMARK_PROTOCOL.md) take precedence. The default case counts and test results of the old version described here do not represent the results of the currently distributed version.

# Selection Rationale for the 10-Case Demo

> Historical document from the initial offline audit. For current results see [README](../../README.md) and [results/SUMMARY.md](../../results/SUMMARY.md).

Curated from the original 48 cases **without changing the source texts or gold labels**. Arranged in the order used for the slide demo. Only the difficulty `group` labels used for the demo were re-classified to match the main verification points.

**Difficulty `group` labels:** clear=explicit, hypothesis=presumed cause, negation=negation / false positive, mixed_status=partially normal, resolved=recovered, out_of_scope=unknown.

**Purpose:** Compare negation, temporal sequence, confirmed vs. suspected causes, and unidentifiable targets under the same rules. Not for use in actual safety decisions on power equipment.

| Order | ID | Equipment | Event | Cause status | Points to watch |
|---:|---|---|---|---|---|
| 1 | PS-009 | Circuit breaker | Operation failure | Confirmed | Warm-up: equipment type, observed anomaly, and confirmed cause are all stated explicitly |
| 2 | PS-001 | Transformer | Heat | Suspected | Does it avoid mistaking "suspected" for a confirmed cause? |
| 3 | PS-002 | Transformer | Oil leak | Unknown | Heat is ruled out; an oil leak is actually observed |
| 4 | PS-018 | Meter | Communication | Suspected | Display is normal but communication fails; radio environment is the suspected cause |
| 5 | PS-031 | Switch | Corrosion | Confirmed | Reads corrosion and a confirmed cause (distinguishing observation from causation) |
| 6 | PS-014 | Overhead line | Damage | Unknown | Temperature at the connection is normal; only a crack at another location is abnormal |
| 7 | PS-010 | Switch | No abnormality | Unknown | Previous corrosion has been repaired; currently no abnormality |
| 8 | PS-037 | Meter | Damage | Unknown | Communication has recovered; only enclosure damage remains |
| 9 | PS-044 | Switch | Heat | Confirmed | Communication-loss report was a false positive; the true event is contact heating, and the cause is confirmed |
| 10 | PS-012 | Out of scope / unknown | Other / unknown | Unknown | Input identifies neither the target nor the event; does it avoid guessing? |

## Source Texts and Gold Labels

### 1. PS-009 — Warm-up: equipment type, observed anomaly, and confirmed cause are all stated explicitly

> The VCB failed to operate after the open command was issued. Investigation found that a break in the control circuit fuse was the cause.

Gold: `{"equipment": "breaker", "observation": "operation_failure", "cause_status": "confirmed"}`

### 2. PS-001 — Does it avoid mistaking "suspected" for a confirmed cause?

> Local temperature rise was confirmed on the outer case of a distribution transformer. Increased contact resistance at the terminal is suspected, but the cause is not yet confirmed.

Gold: `{"equipment": "transformer", "observation": "heat", "cause_status": "suspected"}`

### 3. PS-002 — Heat is ruled out; an oil leak is actually observed

> No heating was confirmed on the distribution transformer, but an oil leak was confirmed at the bottom of the tank. The cause has not been identified.

Gold: `{"equipment": "transformer", "observation": "oil_leak", "cause_status": "unknown"}`

### 4. PS-018 — Display is normal but communication fails; radio environment is the suspected cause

> The meter's device display is normal, but communication with the upper-level system failed. Influence of the surrounding radio environment is suspected.

Gold: `{"equipment": "meter", "observation": "communication", "cause_status": "suspected"}`

### 5. PS-031 — Reads corrosion and a confirmed cause (distinguishing observation from causation)

> Significant rust was confirmed on the switch enclosure. Investigation confirmed that water retention caused by a blocked drain hole is the cause.

Gold: `{"equipment": "switch", "observation": "corrosion", "cause_status": "confirmed"}`

### 6. PS-014 — Temperature at the connection is normal; only a crack at another location is abnormal

> The temperature at the overhead line connection is normal. However, a crack was found on the conductor surface at a different location. Cause unknown.

Gold: `{"equipment": "overhead_line", "observation": "damage", "cause_status": "unknown"}`

### 7. PS-010 — Previous corrosion has been repaired; currently no abnormality

> The corrosion point previously noted on the switch has been repaired. No new abnormality was found in this patrol.

Gold: `{"equipment": "switch", "observation": "normal", "cause_status": "unknown"}`

### 8. PS-037 — Communication has recovered; only enclosure damage remains

> The communication abnormality on the smart meter has been resolved. The only abnormality confirmed this time is a crack in the enclosure.

Gold: `{"equipment": "meter", "observation": "damage", "cause_status": "unknown"}`

### 9. PS-044 — Communication-loss report was a false positive; the true event is contact heating, and the cause is confirmed

> The report of a communication dropout on the switch was a false positive. On the other hand, heat was found at the contacts, and poor contact was confirmed as the cause.

Gold: `{"equipment": "switch", "observation": "heat", "cause_status": "confirmed"}`

### 10. PS-012 — Input identifies neither the target nor the event; does it avoid guessing?

> The patrol report only says "Equipment A looks unusual"; neither the target equipment nor the abnormality can be identified.

Gold: `{"equipment": "other", "observation": "other", "cause_status": "unknown"}`

## Notes for Comparison in the Demo

- The 10 synthetic cases alone cannot statistically establish differences in accuracy, speed, or calibration.
- If prompts or models were adjusted while looking at these same examples, do not claim final performance based on these 10 cases.
- All methods use the same source texts and option definitions. However, differences in model-specific API implementation and execution environment remain.
- The Responses API side provides no candidate probabilities, so ECE and Brier are not directly compared with the other methods.
- Live execution is unverified. Actual API/model operation and usage permissions must be confirmed in your environment.

## Regenerating the Data

```bash
python -m scripts.make_demo_dataset
python -m scripts.verify_dataset
```
