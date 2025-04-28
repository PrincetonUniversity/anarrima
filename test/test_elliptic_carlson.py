import anarrima.elliptic.carlson as carlson
from pytest import approx
import jax.numpy as jnp
jnp.set_printoptions(20)

# Test RF
x0 = 1.
y0 = 2.
z0 = 3.

def test_infinity_rf():
    ans = carlson.rf(0., 0., 1.)
    assert jnp.isinf(ans)

def test_permutations_rf():
    rf = carlson.rf
    w0 = rf(x0, y0, z0)
    w1 = rf(y0, z0, x0)
    w2 = rf(z0, x0, y0)
    assert w0 == w1
    assert w1 == w2

# Test RD
def test_permutations_rd():
    rd = carlson.rd
    w0 = rd(x0, y0, z0)
    w1 = rd(y0, x0, z0)
    assert w0 == approx(w1, rel=2e-16)
