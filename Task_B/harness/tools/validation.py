"""Provider-neutral validation for tool-call arguments."""

from __future__ import annotations

from typing import Any

from ..errors import ErrorCategory, HarnessError


class ToolValidator:
    """Validate the JSON-schema subset used by deterministic harness tools."""

    _TYPE_MAP = {"string": str, "integer": int, "boolean": bool, "object": dict}

    def validate(self, schema: dict[str, Any], arguments: Any) -> None:
        if not isinstance(arguments, dict):
            raise HarnessError("INVALID_TOOL_ARGUMENTS", ErrorCategory.PROTOCOL, "tool arguments must be an object", True)
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        unknown = [name for name in arguments if name not in properties]
        if missing or (unknown and schema.get("additionalProperties") is False):
            raise HarnessError(
                "INVALID_TOOL_ARGUMENTS",
                ErrorCategory.PROTOCOL,
                "tool arguments do not match the schema",
                True,
                {"missing": missing, "unknown": unknown},
            )
        for name, value in arguments.items():
            rule = properties.get(name, {})
            expected = self._TYPE_MAP.get(rule.get("type"))
            if expected and (not isinstance(value, expected) or expected is int and isinstance(value, bool)):
                raise HarnessError("INVALID_TOOL_ARGUMENTS", ErrorCategory.PROTOCOL, f"argument {name!r} has the wrong type", True)
            if isinstance(value, (int, float)):
                if "minimum" in rule and value < rule["minimum"]:
                    raise HarnessError("INVALID_TOOL_ARGUMENTS", ErrorCategory.PROTOCOL, f"argument {name!r} is below minimum", True)
                if "maximum" in rule and value > rule["maximum"]:
                    raise HarnessError("INVALID_TOOL_ARGUMENTS", ErrorCategory.PROTOCOL, f"argument {name!r} exceeds maximum", True)