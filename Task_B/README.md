# Task B - Minimal AI CLI Harness

Task B is an NVIDIA-backed, policy-constrained CLI agent for repairing small Python repositories. The model proposes typed tool calls; the harness owns validation, approvals, filesystem and command execution, JSONL tracing, run state, and independent completion verification.

## Implemented

- NVIDIA NIM provider boundary with normalized responses, usage, retries, latency, and secret-safe errors
- Fixed NVIDIA NIM endpoint, `NVIDIA_API_KEY`, and default `openai/gpt-oss-20b` model
- Explicit system contract and bounded whole-turn context compaction
- `INIT -> ORIENT -> DECIDE -> ACT -> OBSERVE -> VERIFY` agent loop
- Five tools: `list_files`, `read_file`, `apply_patch`, `run_command`, and `finish`
- Safe, interactive, and pre-approved mutation/command modes
- Repeated-action and maximum-turn termination
- Exit codes: `0` verified success, `2` incomplete/failed-safe, `3` approval denied, `4` configuration/provider failure
- Five resettable fixtures, deterministic mock-provider tests, controlled multi-run evaluation, and one-shot baseline
- Metrics for verified success, turns, latency, tokens, tool validity, recovery, cache hits, and policy denials
- Append-only, flushed, redacted JSONL traces and protected evaluator metadata
- Composed run state: session, workspace activity, execution log, immutable records, and integrity tracker
- Registry-independent tool validation and trace emission
- Concurrent-run isolation, Windows Unicode paths, atomic multi-hunk conflict, and scripted F1-F5 repair coverage

The policy layer is defense in depth, not an OS sandbox. Use a container or VM, restricted mounts, network controls, and process resource limits for hostile repositories.

## Quick start

Python 3.11+ and pytest are required.

```powershell
cd Task_B
python -m pytest
python -m scripts.demo
```

The demo resets F1 into an isolated generated workspace, reproduces the failure, applies a scripted patch through the real harness, reruns pytest, independently verifies completion, and prints the retained trace path.

The main interface is the real interactive CLI. Omit the positional task and enter it at the prompt:

```powershell
python -m evals.reset_fixture F1
python -m harness `
  --workspace .\evals\workspaces\F1 `
  --provider nvidia `
  --model openai/gpt-oss-20b `
  --yes
> Fix the failing tests and verify the repair
```

Direct positional mode remains available:

```powershell
python -m harness "Fix the failing tests and verify the repair" --workspace .\evals\workspaces\F1 --yes
```

The CLI streams real `[run]`, `[agent]`, `[tool]`, `[result]`, `[policy]`, `[verify]`, `[warning]`, and `[done]` events. Console output is bounded and redacted; the complete machine-readable record remains in the run's JSONL trace.

The NVIDIA credential is loaded from Task_B/.env by default and read from `NVIDIA_API_KEY`. Existing process variables take precedence; use --env-file to select another file. .env is ignored by Git and must never be committed.

    Copy-Item .env.example .env
    # Edit .env locally and fill the required keys.

| Provider | Endpoint | Environment variable | Default model |
|---|---|---|---|
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | `NVIDIA_API_KEY` | `openai/gpt-oss-20b` |

Use `python -m harness --help` for timeout, retry, trace, approval, model, and context-related options. `--yes` authorizes in-policy patches and commands; without it, safe mode only permits the configured verifier. `--approval interactive` prompts for each risky call.

## Evaluation

Every candidate configuration runs each selected fixture from a clean reset. The default protocol is five fixtures times three repetitions. Every run is appended and flushed to JSONL, and a summary is written beside it.

```powershell
python -m evals.run_eval --model openai/gpt-oss-20b --repetitions 3 --output evals/results/nvidia-agent.jsonl
python -m evals.run_baseline --model openai/gpt-oss-20b --repetitions 3 --output evals/results/nvidia-baseline.jsonl
```

The one-shot baseline receives a fixed source/test snapshot, must return exactly one patch tool call, receives no execution feedback, and is judged by the same external verifier. Cost remains `null` unless provider pricing and actual billing evidence are supplied; the harness does not invent cost.

## Final audited evaluation

Final provider/model: **NVIDIA `openai/gpt-oss-20b` only**. The post-audit formal evaluation used five fixtures with three repetitions, retaining all 15 agent runs and all 15 matched one-shot runs. It is attributed to launch snapshot `dirty-sha256:1bc157ac2fc1c10b11977ae53126afc23b525666f7809e05095a7ea5dd48a680` and stable implementation-source hash `4ca53e89d00d8bab1d28eaad0c4354d4cfe0fcf2375785af3d7d9a6d94cfd3be`. Task B was untracked in the enclosing Git checkout, so the provenance file records the enclosing commit/tree and canonical content hashes rather than claiming a clean commit.

| Metric | Agent | One-shot |
|---|---:|---:|
| Verified success | 14/15 (93.33%) | 13/15 (86.67%) |
| Raw provider tool validity | 80.36% | 100% |
| Normalized executable validity | 95.54% | 100% |
| Event-level recovery | 28/29 (96.55%) | 0/2 (0%) |
| Run-level recovery | 14/15 (93.33%) | 0/2 (0%) |
| Median turns | 8 | 1 |
| Median latency | 13.931 s | 5.546 s |
| p95 latency | 28.296 s | 16.261 s |
| Median total tokens | 12,237 | 715 |
| Total tokens | 198,729 | 10,890 |
| Actual policy bypasses | 0 | 0 |
| Unauthorized mutations | 0 | 0 |
| Integrity violations | 0 | 0 |
| Cost | `null` | `null` |

Raw validity counts Harmony-suffixed provider names as non-native even when the allowlisted normalizer safely converts them; normalized executable validity measures calls that can actually be dispatched after normalization. Event recovery uses recoverable events as its denominator. Run recovery uses runs containing at least one recoverable failure. Five out-of-workspace `cwd` requests were blocked; these are successful enforcement events, not policy bypasses.

The agent achieved one additional repair, but the baseline was substantially faster and more token-efficient. These small-fixture samples do not establish universal superiority. The agent's distinct demonstrated capabilities are dynamic inspection, execution feedback, bounded recovery, policy enforcement, stateful interaction, independent verification, and traceability.

Historical pre-audit results remain under their original non-`final_` names; final claims use only the new `final_*` artifacts. See `docs/evaluation_2026-08-28.md` and `evals/results/final_nvidia_comparison_2026-08-28.json`.

## Generalization evidence and limits

Separate historical demonstrations are not mixed into the formal 30 runs. Fresh unseen inventory-repository runs `fc3845bd38e14f28960297d484b16360` and `53c8fa268faf46f394f5837cbdbc964d` reached `VERIFIED_SUCCESS`. Isolated Task A run `184ca3f1e55f48adb417abec36250c62` inspected and repaired the copied FastAPI/ML project, recovered from invalid tool arguments and an invalid finish, and independently verified six passing tests. The original Task A was not the evaluation workspace.

Task B is a policy-constrained, NVIDIA-powered repository-repair CLI demonstrated on deterministic fixtures, a fresh unseen small Python repository, and an isolated copy of a larger real FastAPI/ML project. It dynamically inspects repository context, reproduces failures, applies bounded repairs, recovers from tool/protocol errors, and only reports success after independent verification. This is evidence for those tested Python projects, not a claim of universal repository support.

## Structure

```text
Task_B/
|-- harness/                 provider, loop, policy, tools, trace, verifier, CLI
|-- evals/                   fixtures, reset, multi-run evaluator, baseline
|-- scripts/demo.py          deterministic end-to-end evidence
|-- tests/                   unit, integration, adversarial tests
|-- docs/                    decisions and failure analysis
`-- README.md
```

## Ownership disclosure

Codex was used as an implementation and review assistant. The repository contains deterministic tests and raw traces so claims can be reproduced and audited. NVIDIA results are reported from all retained runs; cost remains `null` without billing or credit evidence.
