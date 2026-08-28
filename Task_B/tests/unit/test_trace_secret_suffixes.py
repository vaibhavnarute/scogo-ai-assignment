from harness.trace import sanitize_for_log


def test_secret_key_suffixes_are_redacted_without_hiding_usage_counts():
    sanitized = sanitize_for_log(
        {
            "provider_api_key": "secret-a",
            "client_secret": "secret-b",
            "token": "secret-c",
            "token_usage": {"input_tokens": 12, "cached_tokens": 3},
        }
    )
    assert sanitized["provider_api_key"] == "[REDACTED]"
    assert sanitized["client_secret"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["token_usage"] == {"input_tokens": 12, "cached_tokens": 3}

