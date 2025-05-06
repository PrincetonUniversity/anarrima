import anarrima.elliptic.legendre as legendre
from utils import values_match

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad

import mpmath as mp

import pytest
from pytest import approx
from pytest import mark

INF = jnp.inf
PI2 = jnp.pi/2
PI4 = jnp.pi/4
NAN = jnp.nan

isnan = jnp.isnan
isinf = jnp.isinf

einc = legendre.ellipeinc
finc = legendre.ellipfinc

# mpmath settings
mp.mp.dps = 40

# Map of points to test
##########################################
# NaN            L                      M
# 
# π/2  C --------D-------E-------F
#      |         P       | 
#      |                  \
#      |                   |
# π/4  B         K          G           N
#      |                     \
#  0   A---------J-------I-------H      O
# 
#     -∞        -1   0   1       ∞     NaN
##########################################

pA = (0.0, -INF)
pB = (PI4, -INF)
pC = (PI2, -INF)
pD = (PI2, -1.0)
pE = (PI2, +1.0)
pF = (PI2, +INF)
pG = (PI4, +2.0)
pH = (0.0, +INF)
pI = (0.0, +1.0)
pJ = (0.0, -1.0)
pK = (PI4, -1.0)
pL = (NAN, -1.0)
pM = (NAN, NAN )
pN = (PI4, NAN )
pO = (0.0, NAN )
pP = (PI2 - 1e-4, -1.)

### Tests for ellipeinc

def high_precision_einc(φ, m):
    return float(mp.ellipe(φ, m))

test_cases_e = [
    (*pA, 0.0),
    (*pB, INF), # MMA: ComplexInfinity
    (*pC, INF), # MMA: ComplexInfinity
    (*pD, float(mp.ellipe(-1))),
    (*pE, 1.0),
    (*pF, NAN),
    (*pG, high_precision_einc(*pG)),
    (*pH, NAN), # scipy and mpmath return nan here; Mathematica returns ComplexInfinity
    (*pI, 0.0),
    (*pJ, 0.0),
    (*pK, high_precision_einc(*pK)),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
    (*pP, high_precision_einc(*pP)),
]

@pytest.mark.parametrize("phi, m, expected", test_cases_e)
def test_einc(phi, m, expected):
    if expected is None:
        assert False

    result = einc(phi, m)
    assert values_match(result, expected, rel=1e-15)

test_cases_dedφ = [
    (*pA, NAN), # Indeterminate
    (*pB, INF), # MMA: ComplexInfinity
    (*pC, INF), # MMA: ComplexInfinity
    (*pD, jnp.sqrt(2)),
    (*pE, 0.0),
    (*pF, NAN), # MMA: i ∞
    pytest.param(*pG, None, marks=mark.skip("∇ ~ sqrt(distance from edge) so O(e-8) instead of 0.0")),
    (*pH, NAN), # MMA: indet
    (*pI, 1.0),
    (*pJ, 1.0),
    (*pK, jnp.sqrt(3./2)),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
    (*pP, float(mp.sqrt(1 + mp.cos(1e-4)**2))),
]

@pytest.mark.parametrize("phi, m, expected", test_cases_dedφ)
def test_deinc_dφ(phi, m, expected):
    if expected is None:
        assert False

    result = grad(einc, argnums=0)(phi, m)
    assert values_match(result, expected, rel=1e-15)

def test_deinc_dφ_pG():
    de_dφ = grad(einc, argnums=0)(*pG)
    assert 0.0 == approx(de_dφ, abs=3e-8)

def high_precision_dedm(φ, m):
    dedm = (mp.ellipe(φ, m) - mp.ellipf(φ, m))/(2 * m)
    return float(dedm)

test_cases_dedm = [
    (*pA, 0.0),
    (*pB, 0.0), # MMA: ComplexInfinity
    (*pC, 0.0), # MMA: ComplexInfinity
    (*pD, high_precision_dedm(*pD)),
    pytest.param(*pE, None, marks=mark.skip("Approaches -∞; precision-dependent")),
    (*pF, NAN), # unclear; could be zero in absolute magnitude but approach is complex
    pytest.param(*pG, None, marks=mark.skip("∇ ~ sqrt(distance from edge) so O(e-8)")),
    # mpmath return nan from ∞ * 0; Mathematica says Indeterminate...; 
    (*pH, NAN),
    (*pI, high_precision_dedm(*pI)),
    (*pJ, high_precision_dedm(*pJ)),
    (*pK, high_precision_dedm(*pK)),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
    (*pP, high_precision_dedm(*pP)),
]

@pytest.mark.parametrize("phi, m, expected", test_cases_dedm)
def test_deinc_dm(phi, m, expected):
    if expected is None:
        assert False

    result = grad(einc, argnums=1)(phi, m)
    assert values_match(result, expected, rel=1e-15)

def test_deinc_dm_pE():
    de_dm = grad(einc, argnums=1)(*pE)
    assert de_dm < -18.0 # typical value at this precision

def test_deinc_dm_pG():
    de_dm = grad(einc, argnums=1)(*pG)
    high_prec = high_precision_dedm(*pG)
    assert de_dm == approx(high_prec, rel=1e-8)
