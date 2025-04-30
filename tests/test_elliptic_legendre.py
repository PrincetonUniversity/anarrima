import anarrima.elliptic.legendre as legendre
import pytest
from pytest import approx
import jax.numpy as jnp
from jax import grad

INF = jnp.inf
isnan = jnp.isnan
isneginf = jnp.isneginf
nan = jnp.nan

einc = legendre.ellipeinc
finc = legendre.ellipfinc

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

## Gradients of ellipf
def test_gradellipf_phi_at_zero():
    g_f1 = grad(finc)(0., 0.1)
    assert g_f1 == 1.0

### Test ellipeinc
@pytest.mark.skip(reason="Not testing E")
def test_ellipe_phi_zero():
    e1 = einc(0., m0)
    assert e1 == 0.

@pytest.mark.skip(reason="Not testing E")
def test_ellipe_m_neginf():
    e1 = einc(φ0, -INF)
    assert isneginf(e1)

@pytest.mark.skip(reason="Not testing E")
def test_ellipe_m_zero():
    e1 = einc(φ0, 0.)
    assert e1 == φ0

@pytest.mark.skip(reason="Not testing E")
def test_ellipe_phi_nan():
    e1 = einc(nan, m0)
    assert isnan(e1)

@pytest.mark.skip(reason="Not testing E")
def test_ellipe_m_nan():
    e1 = einc(φ0, nan)
    assert isnan(e1)

def test_fused_form_equality():
    f1 = finc(φ0, m0)
    e1 = einc(φ0, m0)
    f2, e2 = legendre.ellip_finc_einc_fused(φ0, m0)
    assert f1 == f2
    assert e1 == e2

