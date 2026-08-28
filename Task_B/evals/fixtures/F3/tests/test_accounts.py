import pytest

from accounts import normalize_username


def test_empty_username_uses_public_value_error_contract():
    with pytest.raises(ValueError, match="required"):
        normalize_username("")


def test_username_is_normalized():
    assert normalize_username("  Alice  ") == "alice"

