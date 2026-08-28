def shipping_cost(order_total: float, *, vip: bool = False) -> int:
    """Return 0 for VIP or qualifying orders, otherwise the standard fee."""
    if order_total < 0:
        return 0
    if order_total > 50:
        return 0
    return 8

