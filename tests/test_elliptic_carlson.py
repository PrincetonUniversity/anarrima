import pytest
from pytest import approx

import anarrima.elliptic.carlson as carlson

rf = carlson.rf
rd = carlson.rd

# Test RF
x0 = 1.0
y0 = 2.0
z0 = 3.0


@pytest.mark.skip(reason="This cannot happen in anarrima geometry")
def test_infinity_rf():
    ans = rf(0.0, 0.0, 1.0)
    assert jnp.isinf(ans)


def test_permutations_rf():
    w0 = rf(x0, y0, z0)
    w1 = rf(y0, z0, x0)
    w2 = rf(z0, x0, y0)
    assert w0 == w1
    assert w1 == w2


# Test RD
def test_permutations_rd():
    w0 = rd(x0, y0, z0)
    w1 = rd(y0, x0, z0)
    assert w0 == approx(w1, rel=2e-16)
