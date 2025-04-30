import anarrima.elliptic.legendre as legendre
import pytest
from pytest import approx
import jax.numpy as jnp
from jax import grad

import mpmath as mp
from mpmath import almosteq, mpf

INF = jnp.inf
isnan = jnp.isnan
isinf = jnp.isinf
isneginf = jnp.isneginf
nan = jnp.nan

einc = legendre.ellipeinc
finc = legendre.ellipfinc

# mpmath settings
mp.mp.dps = 40

# Test RF
φ0 = 0.5
m0 = -1.0

### Ellipfinc tests
def test_ellipf_antisymmetry():
    f1 = finc(φ0, m0)
    f2 = finc(-φ0, m0)
    assert f1 == -f2

def test_ellipf_phi_zero():
    f1 = finc(0., m0)
    assert f1 == 0.

def test_ellipf_m_neginf():
    f1 = finc(φ0, -INF)
    assert f1 == 0.

def test_ellipf_m_zero():
    f1 = finc(φ0, 0.)
    assert f1 == φ0

def test_ellipf_phi_zero_m_zero():
    f1 = finc(0., 0.)
    assert f1 == 0.

def test_ellipf_phi_zero_m_neginf():
    f1 = finc(0., -INF)
    assert f1 == 0.

def test_ellipf_phi_nan():
    f1 = finc(nan, m0)
    assert isnan(f1)

def test_ellipf_m_nan():
    f1 = finc(φ0, nan)
    assert isnan(f1)

def test_ellipf_phi_nan_m_nan():
    f1 = finc(nan, nan)
    assert isnan(f1)

def test_ellipf_high_precision():
    f1 = float(finc(φ0, m0))
    hp = mp.ellipf(mpf(φ0), mpf(m0))
    assert almosteq(f1, hp, rel_eps=1e-16)


## Gradients of ellipf

def high_precison_dfdm(φ, m):
    φ = mpf(φ)
    m = mpf(m)
    sinφ = mp.sin(φ)
    sin_sq_φ = sinφ * sinφ
    sin_cu_φ = sinφ * sin_sq_φ
    x = 1 - sin_sq_φ
    y = 1 - m * sin_sq_φ
    z = 1

    # compute dF/dm
    # note the specific argument order!
    d_ellipf_dm = sin_cu_φ * mp.elliprd(z, x, y) / 6
    return d_ellipf_dm

def test_gradellipf_typical():
    g_f1 = float(grad(finc, argnums=1)(φ0, m0))
    g_hp = high_precison_dfdm(φ0, m0)
    assert almosteq(g_f1, g_hp, rel_eps=1e-17)

def test_gradellipf_near_large_phi():
    φ = jnp.pi/2 - 1e-4
    m = -1.
    g_f1 = float(grad(finc, argnums=1)(φ, m))
    g_hp = high_precison_dfdm(φ, m)
    assert almosteq(g_f1, g_hp, rel_eps=1e-17)

@pytest.mark.skip(reason="gradient jvp not yet defined for this special case")
def test_gradellipf_phi_at_zero():
    g_f1 = grad(finc)(0., 0.1)
    assert g_f1 == 1.0

### Test ellipeinc
def test_ellipe_phi_zero():
    e1 = einc(0., m0)
    assert e1 == 0.

def test_ellipe_m_posinf():
    e1 = einc(φ0, INF)
    assert isnan(e1)

def test_ellipe_m_neginf():
    e1 = einc(φ0, -INF)
    assert isinf(e1)

def test_ellipe_phi_zero_m_neginf():
    e1 = einc(0., -INF)
    assert isinf(e1)

def test_ellipe_m_zero():
    e1 = einc(φ0, 0.)
    assert e1 == φ0

def test_ellipe_phi_nan():
    e1 = einc(nan, m0)
    assert isnan(e1)

def test_ellipe_m_nan():
    e1 = einc(φ0, nan)
    assert isnan(e1)

def test_ellipe_phi_nan_m_nan():
    e1 = einc(nan, nan)
    assert isnan(e1)

def test_fused_form_equality():
    f1 = finc(φ0, m0)
    e1 = einc(φ0, m0)
    f2, e2 = legendre.ellip_finc_einc_fused(φ0, m0)
    assert f1 == f2
    assert e1 == e2

