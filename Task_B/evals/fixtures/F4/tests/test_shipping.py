import pytest

from shipping import shipping_cost


def test_threshold_is_inclusive():
    assert shipping_cost(50) == 0


def test_vip_gets_free_shipping_below_threshold():
    assert shipping_cost(10, vip=True) == 0


def test_negative_total_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        shipping_cost(-1)


def test_regular_order_pays_fee():
    assert shipping_cost(10) == 8

