import pytest

from buggy_math import add, divide, multiply, power, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2
    assert subtract(3, 5) == -2
    assert subtract(0, 0) == 0


def test_multiply():
    assert multiply(2, 3) == 6
    assert multiply(0, 5) == 0
    assert multiply(-2, 3) == -6


def test_divide():
    assert divide(6, 3) == 2.0
    assert divide(5, 2) == 2.5
    assert divide(-6, 3) == -2.0
    with pytest.raises(ValueError):
        divide(5, 0)


def test_power():
    assert power(2, 3) == 8
    assert power(5, 0) == 1
    assert power(2, -1) == 0.5


