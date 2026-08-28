# Task B implementation decisions

Meaningful changes to the planned tool surface, safety boundary, provider abstraction, verification semantics, context policy, integrity rules, trace format, or CLI behavior are recorded here.

## D-01 - Reliability foundation moved earlier

The brief's first-pass checkpoint required trace and verifier foundations before the provider loop, while the broader phase table placed them later. They were implemented in the deterministic substrate. This changed sequencing, not the architecture or contract.

## D-02 - Unified-diff patching only

`apply_patch` supports standard unified-diff hunks against one existing workspace file. No unconstrained overwrite or controlled-edit fallback was added. Full-file headers are accepted, but policy still selects and validates the explicit `path` argument.

## D-03 - NVIDIA NIM chat-completions boundary

The live adapter targets NVIDIA NIM only, with a fixed endpoint and `NVIDIA_API_KEY`. The model remains configurable and defaults to NVIDIA catalog identifier `openai/gpt-oss-20b`. The provider interface remains separate from tool and agent contracts, and the adapter uses the Python standard library without an SDK dependency.

## D-04 - Whole-turn context compaction

Context limits retain the system/task contract and compact only complete assistant/tool exchange batches. Tool results are never separated from the model calls that produced them. This makes truncation deterministic and avoids protocol-invalid orphan messages.

## D-05 - Completion belongs to the verifier

A model `finish` call is only a request. It succeeds only after the exact configured verifier passed after the latest mutation and protected paths still match their initial hashes. Command-side repository mutations are detected as integrity violations because `apply_patch` is the sole authorized editing tool.

## D-06 - Controlled evaluation and cost policy

Agent-loop and one-shot configurations use the same fixture revisions, clean reset, and external verifier. Defaults are five fixtures times three repetitions, with every record flushed to JSONL. Cost is `null` unless pricing inputs and billing evidence exist; token usage alone is not converted into an invented charge.

## HR-01 - Phase 1 hardening review

A review found gaps in command-side mutation tracking, fixture-metadata protection, Windows command parsing, process-tree termination, token-usage sanitization, bounded listing traversal, normalized error detail, and full unified-diff compatibility. Each was corrected and covered by regression tests.

## D-07 - Composed state with compatibility facade

`RunState` now composes `SessionState`, `WorkspaceActivity`, `ExecutionLog`, and `IntegrityTracker`; immutable command/mutation/verification records live separately. Read/write compatibility properties retain the existing tool and evaluator contract while component ownership and independent testing are explicit.

## D-08 - Registry responsibility split

JSON-schema-subset validation moved to `ToolValidator`, and dispatch trace emission moved to `ToolTraceEmitter`. `ToolRegistry` now owns registration, lookup, repeated-action accounting, execution, and the exception boundary. Existing event names and error normalization remain stable.

## HR-02 - Isolation and fixture coverage

Regression coverage now includes simultaneous agents with distinct state/workspaces/traces, Windows Unicode filenames through list/read/patch, atomic failure of a later conflicting hunk, and verified scripted agent repairs for F2-F5 in addition to F1.
## D-09 - Local credential loading

The CLI and evaluation provider factory load Task_B/.env with a strict standard-library parser before resolving `NVIDIA_API_KEY`. Existing process variables win unless override is explicitly requested by code. Empty values are ignored, parser errors expose only path and line number, .env is Git-ignored, and .env.example contains names only.
## LV-02 - Live NVIDIA tool calling

The NVIDIA default is `openai/gpt-oss-20b`. After a repository-free protocol probe, a controlled live F4 run reached harness-owned `VERIFIED_SUCCESS`: it recovered from invalid tool arguments, applied one policy-bounded source mutation, and passed internal and independent verification. The retained trace was followed by a formal NVIDIA-only evaluation of 15 agent-loop runs and 15 matched baselines; no cross-provider quality claim is made.
