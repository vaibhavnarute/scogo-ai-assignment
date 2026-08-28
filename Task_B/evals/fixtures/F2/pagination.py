def page_count(item_count: int, page_size: int) -> int:
    """Return the number of pages needed for item_count items."""
    if item_count == 0:
        return 0
    return item_count // page_size + 1

