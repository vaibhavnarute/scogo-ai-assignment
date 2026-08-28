# Failure analysis and measured limitations

## Expected runtime failure classes

- Configuration/authentication: fail before tool execution with a normalized provider/configuration result.
- Rate limit or transient 5xx/network error: retry within the configured bound, record retry count and latency, then fail safely if exhausted.
- Malformed or unknown tool call: return recoverable protocol evidence to the model; repeated identical actions terminate.
- Failed patch or command: preserve the tool error/exit evidence for a later repair turn.
- Premature finish: reject unless the latest exact verifier passed after the latest mutation and protected paths remain intact.
- Timeout: terminate the process tree, retain bounded output, and return normalized timeout evidence.
- Trace failure: terminal, because an untraceable run cannot satisfy the assignment contract.

## Measurement limitations

Evaluation output reports all attempted runs and fixture revision hashes. Recovery is counted only when a recoverable tool failure is followed by a later successful tool completion in the same trace. This is a transparent operational proxy; it does not prove that the later action semantically corrected the original mistake.

Token counts depend on provider-reported usage. Cost remains unknown without documented price inputs and billing evidence. Wall-clock results include local tool time and provider latency and should be compared only under a controlled environment.

The command policy is not a containment boundary. A production deployment still requires process, filesystem, resource, and network isolation outside this Python process.
## Live NVIDIA protocol success

On 2026-08-28, NVIDIA model openai/gpt-oss-20b completed a repository-free synthetic protocol probe in approximately 1.4 seconds. It returned finish_reason=tool_calls and exactly one valid ping call with empty arguments. Provider-reported usage was 134 input tokens, 67 output tokens, and 201 total tokens.

This establishes working authentication, endpoint access, response normalization, usage normalization, and live tool-call compatibility. It does not establish repository inspection quality, patch correctness, recovery behavior, policy compliance under live model decisions, or verified repair success.
## Live NVIDIA F4 repair success

On 2026-08-28, NVIDIA `openai/gpt-oss-20b` completed a live F4 repair with `VERIFIED_SUCCESS` in 10 turns and 17.249 seconds of traced elapsed time. It listed the repository, read `shipping.py` and the protected test, ran the verifier to observe three failures, generated and applied a repair, and reached four passing tests. An independent post-run `python -m pytest -q` also reported four passing tests.

Recovery occurred inside the successful trace: an empty-path `list_files` call failed with `INVALID_PATH` and was corrected, then a malformed `apply_patch` call failed schema validation and was followed by a successful patch. The trace contains one `file.modified` event for `shipping.py`, no policy denials or unauthorized command mutations, two explicitly allowed configured-verifier commands, and matching SHA-256 hashes for protected tests and fixture metadata. Provider-reported usage was 17,878 input and 1,404 output tokens (19,282 total).

This is evidence for one controlled NVIDIA fixture run; the formal NVIDIA evaluation is reported separately in `docs/evaluation_2026-08-28.md`.

## Final audited formal failures

The post-audit formal evaluation retained three unsuccessful attempts. Agent run `25387ee244a34bc8bfdaa5f6935cc995` failed safely on F3 repetition 3 after two malformed patch envelopes and repeated-action termination. Baseline runs `f831a7a89d4c467ab68852d23cca3731` (F2 repetition 1) and `4763ce1fdc8849dc83579ef35a12f839` (F3 repetition 1) each produced one invalid patch envelope and had no recovery opportunity by design. Trace evidence supports `MODEL` as the root-cause category for all three; no provider, policy, integrity, fixture, or unauthorized-mutation failure reached success.
