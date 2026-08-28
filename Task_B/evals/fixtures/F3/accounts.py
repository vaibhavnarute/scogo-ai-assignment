def normalize_username(username: str) -> str:
    """Validate and normalize a public username."""
    if not username:
        raise TypeError("username is required")
    if len(username) < 3:
        raise ValueError("username must contain at least three characters")
    return username.strip().lower()

