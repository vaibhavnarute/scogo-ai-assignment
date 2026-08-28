from discounts import apply_discount


def test_discount_reduces_price():
    assert apply_discount(100, 15) == 85

