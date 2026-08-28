def apply_discount(price: float, percent: float) -> float:
    """Apply a percentage discount to price."""
    return price * (1 + percent / 100)

