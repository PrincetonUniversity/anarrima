import anarrima.elliptic.legendre as legendre
import pytest
from pytest import approx
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

import mpmath as mp
from mpmath import almosteq, mpf

INF = jnp.inf
ONE = 1.0
ZER = 0.0
PI2 = jnp.pi/2
PI4 = jnp.pi/4
NAN = jnp.nan

isnan = jnp.isnan
isinf = jnp.isinf
isneginf = jnp.isneginf

einc = legendre.ellipeinc
finc = legendre.ellipfinc

# mpmath settings
mp.mp.dps = 40

# NaN            L                      M
# 
# π/2  C --------D-------E-------F
#      |                  |
# π/4  B         K         G            N
#      |                    \
#  0   A---------J-------I-------H      O
# 
#     -∞        -1   0   1       ∞     NaN

pA = (0.0, -INF)
pB = (PI4, -INF)
pC = (PI2, -INF)
pD = (PI2, -ONE)
pE = (PI2, +ONE)
pF = (PI2, +INF)
pG = (PI4, +2.0)
pH = (0.0, +INF)
pI = (0.0, +1. )
pJ = (0.0, -1. )
pK = (PI4, -1. )
pL = (NAN, -1. )
pM = (NAN, NAN )
pN = (PI4, NAN )
pO = (0.0, NAN )

# Define test cases with points and expected values
test_cases_f = [
    (*pA, 0.0),
    (*pB, 0.0),
    (*pC, 0.0),
    (*pD, float(mp.ellipk(-1))),
    # spike (or fin, at φ > π/2) up to ∞
    # result depends on precision, in mpmath
    (*pE, None),
    (*pF, NAN),
    (*pG, float(mp.ellipf(PI4, +2))),
    (*pH, NAN), # Mathematica returns 0
    (*pI, 0.0),
    (*pJ, 0.0),
    (*pK, float(mp.ellipf(PI4, -1))),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
]

@pytest.mark.parametrize("phi, m, expected", test_cases_f)
def test_finc(phi, m, expected):
    if expected is None:
        return

    result = finc(phi, m)
    assert values_match(result, expected)

def test_finc_pE():
    assert finc(*pE) > 38.025

# Helper function to check if values match, handling special cases
def values_match(a, b):
    if isnan(a) and isnan(b):
        return True
    elif isinf(a) and isinf(b):
        return jnp.sign(a) == jnp.sign(b)
    else:
        # For finite values, use approximate comparison
        return a == approx(b, rel=1e-6)
