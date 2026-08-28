from calculator import add


def test_adds_positive_numbers():
    assert add(7, 5) == 12


def test_adds_negative_numbers():
    assert add(-2, -3) == -5

