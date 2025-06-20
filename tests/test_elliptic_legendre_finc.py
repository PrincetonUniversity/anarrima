import jax
from utils import values_match

import anarrima.elliptic.legendre as legendre

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import mpmath as mp
import pytest
from jax import grad
from pytest import approx, mark

INF = jnp.inf
PI2 = jnp.pi / 2
PI4 = jnp.pi / 4
NAN = jnp.nan

isnan = jnp.isnan
isinf = jnp.isinf

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
pM = (NAN, NAN)
pN = (PI4, NAN)
pO = (0.0, NAN)
pP = (PI2 - 1e-4, -1.0)

### Tests for ellipfinc


def high_precision_finc(φ, m):
    return float(mp.ellipf(φ, m))


# Define test cases with points and expected values
test_cases_f = [
    (*pA, 0.0),
    (*pB, 0.0),
    (*pC, 0.0),
    (*pD, float(mp.ellipk(-1))),
    # spike (or fin, at φ > π/2) up to ∞
    # result depends on precision, in mpmath
    pytest.param(*pE, None, marks=mark.skip("At pole")),
    (*pF, NAN),
    # slightly-off sin(φ) & catastrophic subtraction make
    # y wrong by 1e-16 and the result is off by 1e-8
    pytest.param(*pG, None, marks=mark.skip("Loss of precision here")),
    # Mathematica returns 0
    pytest.param(*pH, None, marks=mark.skip("0 or Indeterminate")),
    (*pI, 0.0),
    (*pJ, 0.0),
    (*pK, float(mp.ellipf(*pK))),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
    (*pP, high_precision_finc(*pP)),
]


@pytest.mark.parametrize("phi, m, expected", test_cases_f)
def test_finc(phi, m, expected):
    if expected is None:
        assert False

    result = finc(phi, m)
    assert values_match(result, expected, rel=1e-15)


def test_finc_pE():
    assert finc(*pE) > 38.025


def test_finc_pG():
    result = finc(*pG)
    expected = float(mp.ellipf(*pG))
    assert result == approx(expected, rel=3e-8)


@pytest.mark.parametrize("phi, m, _", test_cases_f)
def test_ellipf_antisymmetry(phi, m, _):
    pos = finc(phi, m)
    neg = finc(-phi, m)
    assert values_match(pos, -neg)


# Define test cases with points and expected values
test_cases_dfdφ = [
    # kind of intederminate;
    # could be 0 or 1 depending on interpretation
    (*pA, 0.0),
    (*pB, 0.0),
    (*pC, 0.0),
    (*pD, 1 / jnp.sqrt(2)),
    # spike (or fin, at φ > π/2) up to ∞
    # result depends on precision, in mpmath
    pytest.param(*pE, None, marks=mark.skip("At pole")),
    (*pF, NAN),
    pytest.param(*pG, None, marks=mark.skip("See note on point G")),
    (*pH, NAN),
    (*pI, 1.0),
    (*pJ, 1.0),
    (*pK, jnp.sqrt(2 / 3)),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
]


@pytest.mark.parametrize("phi, m, expected", test_cases_dfdφ)
def test_dfinc_dphi(phi, m, expected):
    if expected is None:
        assert False

    result = grad(finc)(phi, m)
    assert values_match(result, expected)


def high_precison_dfdm(φ, m):
    sinφ = mp.sin(φ)
    sin_sq_φ = sinφ * sinφ
    sin_cu_φ = sinφ * sin_sq_φ
    x = mp.cos(φ) ** 2
    y = 1 - m * sin_sq_φ
    z = 1

    # compute dF/dm
    # note the specific argument order!
    d_ellipf_dm = sin_cu_φ * mp.elliprd(z, x, y) / 6
    return float(d_ellipf_dm)


test_cases_dfdm = [
    (*pA, 0.0),
    (*pB, 0.0),
    (*pC, 0.0),
    (*pD, high_precison_dfdm(*pD)),
    (*pE, jnp.inf),
    (*pF, NAN),
    pytest.param(*pG, None, marks=mark.skip("goes to ∞; see note above")),
    pytest.param(*pH, None, marks=mark.skip("indeterminate / zero")),
    (*pI, 0.0),
    (*pJ, 0.0),
    (*pK, high_precison_dfdm(*pK)),
    (*pL, NAN),
    (*pM, NAN),
    (*pN, NAN),
    (*pO, NAN),
    (*pP, high_precison_dfdm(*pP)),
]


@pytest.mark.parametrize("phi, m, expected", test_cases_dfdm)
def test_dfinc_dm(phi, m, expected):
    if expected is None:
        assert False

    result = grad(finc, argnums=1)(phi, m)
    assert values_match(result, expected)
