"""Completion request tool; success remains verifier-owned."""

from __future__ import annotations

from ..verify import Verifier
from .base import BaseTool, RiskCategory, ToolCall, ToolContext, ToolResult


class FinishTool(BaseTool):
    name = "finish"
    description = "Request completion after current evidence supports a verified repair."
    risk = RiskCategory.COMPLETION
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}, "evidence": {"type": "string"}},
        "required": ["summary", "evidence"],
        "additionalProperties": False,
    }

    def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        if context.tracer:
            context.tracer.record("verification.started", tool_call_id=call.id)
        decision = Verifier(context.config).evaluate_finish(context.state)
        data = {"accepted": decision.accepted, "summary": call.arguments["summary"], "evidence_claim": call.arguments["evidence"], "verifier_reason": decision.reason, "verification_evidence": decision.evidence}
        if context.tracer:
            context.tracer.record("verification.passed" if decision.accepted else "verification.failed", tool_call_id=call.id, error_code=decision.error_code, reason=decision.reason, evidence=decision.evidence)
        if decision.accepted:
            return ToolResult(call.id, self.name, True, data)
        return ToolResult(call.id, self.name, False, data, decision.error_code, decision.reason, True)

