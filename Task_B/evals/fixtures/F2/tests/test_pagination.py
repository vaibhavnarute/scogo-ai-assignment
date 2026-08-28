from pagination import page_count


def test_empty_collection_has_no_pages():
    assert page_count(0, 10) == 0


def test_partial_page_is_counted():
    assert page_count(11, 10) == 2


def test_exact_page_boundary_has_no_extra_page():
    assert page_count(20, 10) == 2

