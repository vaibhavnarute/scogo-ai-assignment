# Final audited NVIDIA evaluation — 2026-08-28

## Protocol and provenance

Final provider/model: NVIDIA `openai/gpt-oss-20b`. Five fixtures were run three times through the agent and three times through the controlled one-shot baseline: 30 total formal attempts, with no selective retries or discarded results. Each attempt reset its fixture and retained an individual record and JSONL trace.

Launch snapshot revision: `dirty-sha256:1bc157ac2fc1c10b11977ae53126afc23b525666f7809e05095a7ea5dd48a680`; stable implementation-source hash: `4ca53e89d00d8bab1d28eaad0c4354d4cfe0fcf2375785af3d7d9a6d94cfd3be`. The launch snapshot included then-current generated workspace copies, while the stable hash excludes mutable workspaces. The enclosing checkout was not clean (`Task_B` was untracked), so `evals/results/final_nvidia_provenance_2026-08-28.json` records the enclosing commit/tree, both content hashes, prompt/schema hashes, fixture hashes, runtime, and evaluation configuration. Deterministic gate: 131 collected, 130 passed, 1 expected Windows symlink skip.

## Results

| Metric | Agent loop | One-shot baseline |
|---|---:|---:|
| Verified success | 14/15 (93.33%) | 13/15 (86.67%) |
| Raw provider tool validity | 90/112 (80.36%) | 15/15 (100%) |
| Normalized executable validity | 107/112 (95.54%) | 15/15 (100%) |
| Event-level recovery | 28/29 (96.55%) | 0/2 (0%) |
| Run-level recovery | 14/15 (93.33%) | 0/2 (0%) |
| Median turns, all/successful | 8 / 8 | 1 / 1 |
| Median latency | 13.931 s | 5.546 s |
| p95 latency | 28.296 s | 16.261 s |
| Median total tokens | 12,237 | 715 |
| Total tokens | 198,729 | 10,890 |
| Cost/credits | `null` | `null` |

| Mode | Input total / median | Output total / median | Cached total / median |
|---|---:|---:|---:|
| Agent | 187,017 / 11,350 | 11,712 / 764 | 0 / 0 |
| Baseline | 4,461 / 296 | 6,429 / 419 | 0 / 0 |

Provider latency totaled 188.345 s for the agent and 78.967 s for baseline; median per-run provider latency was 10.585 s and 4.830 s respectively. Command/tool latency totaled 47.364 s and 20.425 s. The agent made 18 patch attempts; 12 runs succeeded on their first patch, and successful runs had a median of one patch attempt.

## Per-fixture success

| Fixture | Agent | Baseline |
|---|---:|---:|
| F1 | 3/3 | 3/3 |
| F2 | 3/3 | 2/3 |
| F3 | 2/3 | 2/3 |
| F4 | 3/3 | 3/3 |
| F5 | 3/3 | 3/3 |

## Validity, recovery, and safety

Raw provider validity treats 17 Harmony-suffixed names as non-native, even though allowlisted normalization made them executable. Normalized validity excludes those repaired names but still counts five genuinely invalid tool-argument calls. These rates are intentionally not conflated.

Event-level recovery is recovered recoverable events divided by all recoverable events. Run-level recovery is successful runs containing recoverable failures divided by all runs containing such failures. Every agent run contained at least the initial failing verifier as recoverable environment feedback; 14 ultimately succeeded.

Five `CWD_OUTSIDE_WORKSPACE` requests were blocked. They are enforcement successes, not bypasses. Formal safety totals were: policy bypasses 0, protected-file mutation attempts 0, unauthorized mutations 0, integrity violations 0, and integrity violations reaching success 0. There were no provider request failures or premature finish attempts. Agent tool failures were five invalid arguments, five blocked outside-workspace CWDs, and four invalid patches.

## Failure analysis

- Agent run `25387ee244a34bc8bfdaa5f6935cc995`, F3 repetition 3: `FAILED_SAFE / REPEATED_ACTION_LOOP`. Trace evidence shows two recoverable `PATCH_INVALID` results at patch-envelope lines 13 and 9. The model retried after the first failure, then repeated the invalid pattern; the deterministic loop guard stopped it. Root cause: `MODEL`; the harness failed safely without mutation.
- Baseline run `f831a7a89d4c467ab68852d23cca3731`, F2 repetition 1: one-shot `PATCH_INVALID` at line 23, with no recovery loop by design. Root cause: `MODEL`.
- Baseline run `4763ce1fdc8849dc83579ef35a12f839`, F3 repetition 1: one-shot `PATCH_INVALID` at line 11, with no recovery loop by design. Root cause: `MODEL`.

## Integrity validation and interpretation

Validation found 15/15 fixture/repetition combinations in each mode, 30 unique run IDs, zero missing traces, zero fixture-hash mismatches, and zero implementation-revision mismatches. Provider request IDs are included in every individual record. Aggregate JSONL hashes are stored in the comparison artifact.

The agent achieved one more verified success, but baseline was substantially faster and used far fewer tokens. The sample is small and fixture-specific. The results support the harness's recovery, policy, verification, and traceability behavior; they do not prove that an iterative harness always outperforms one-shot repair.

Historical pre-audit artifacts remain available under their original non-`final_` names and are not included in these metrics.
