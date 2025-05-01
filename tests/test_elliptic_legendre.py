import anarrima.elliptic.legendre as legendre
import pytest
from pytest import approx
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

# pA = (0.0, -INF)
# pB = (PI4, -INF)
# pC = (PI2, -INF)
# pD = (PI2, -1.)
# pE = (PI2, +1.)
# pF = (PI2, +INF)
# pG = (PI4, +2)
# pH = (0.0, +INF)
# pI = (0.0, +1.)
# pJ = (0.0, -1.)
# pK = (PI4, -1.)
# pL = (NAN, -1.)
# pM = (NAN, NAN)
# pN = (PI4, NAN)
# pO = (0.0, NAN)

# NaN            L                      M
# 
# π/2  C --------D-------E-------F
#      |                  |
# π/4  B         K         G            N
#      |                    \
#  0   A---------J-------I-------H      O
# 
#     -∞        -1   0   1       ∞     NaN

# # Define test cases with points and expected values
# test_cases = [
#     (0.0, -INF, 0.0), # 'A'
#     (PI4, -INF, None), # 'B'
#     (PI2, -INF, None), # 'C'
#     (PI2, -1., None),  # 'D'
#     (PI2, +1., None),  # 'E'
#     (PI2, +INF,None),  # 'F'
#     (PI4, +2, None),   # 'G'
#     (0.0, +INF, None), # 'H'
#     (0.0, +1., None),  # 'I'
#     (0.0, -1., None),  # 'J'
#     (PI4, -1., None),  # 'K'
#     (NAN, -1., None),  # 'L'
#     (NAN, NAN, None),  # 'M'
#     (PI4, NAN, None),  # 'N'
#     (0.0, NAN, None),  # 'O'
# ]
# 
# @pytest.mark.parameterize("phi,m,expected", test_cases)
# def test_finc(phi, m, expected):
#     if expected is None:
#         return
# 
#     result = finc(phi, m)
#     assert values_match(result, expected)
# 
# # Helper function to check if values match, handling special cases
# def values_match(a, b):
#     if isnan(a) and isnan(b):
#         return True
#     elif isinf(a) and isinf(b):
#         return jnp.sign(a) == jnp.sign(b)
#     else:
#         # For finite values, use approximate comparison
#         return approx(a, b, rel=1e-15)

### Ellipfinc tests
# def test_ellipf_antisymmetry():
#     φ, m = pK
#     f1 = finc(φ, m)
#     f2 = finc(-φ, m)
#     assert f1 == -f2
# 
# def test_ellipf_phi_zero():
#     f1 = finc(0., m0)
#     assert f1 == 0.
# 
# def test_ellipf_m_neginf():
#     f1 = finc(φ0, -INF)
#     assert f1 == 0.
# 
# def test_ellipf_m_zero():
#     f1 = finc(φ0, 0.)
#     assert f1 == φ0
# 
# def test_ellipf_phi_zero_m_zero():
#     f1 = finc(0., 0.)
#     assert f1 == 0.
# 
# def test_ellipf_phi_zero_m_neginf():
#     f1 = finc(0., -INF)
#     assert f1 == 0.
# 
# def test_ellipf_phi_nan():
#     f1 = finc(NAN, m0)
#     assert isnan(f1)
# 
# def test_ellipf_m_nan():
#     f1 = finc(φ0, NAN)
#     assert isnan(f1)
# 
# def test_ellipf_phi_nan_m_nan():
#     f1 = finc(NAN, NAN)
#     assert isnan(f1)
# 
# def test_ellipf_high_precision():
#     f1 = float(finc(φ0, m0))
#     hp = mp.ellipf(mpf(φ0), mpf(m0))
#     assert almosteq(f1, hp, rel_eps=1e-16)


## Gradients of ellipf

# def high_precison_dfdm(φ, m):
#     φ = mpf(φ)
#     m = mpf(m)
#     sinφ = mp.sin(φ)
#     sin_sq_φ = sinφ * sinφ
#     sin_cu_φ = sinφ * sin_sq_φ
#     x = mp.cos(φ)**2
#     y = 1 - m * sin_sq_φ
#     z = 1
# 
#     # compute dF/dm
#     # note the specific argument order!
#     d_ellipf_dm = sin_cu_φ * mp.elliprd(z, x, y) / 6
#     return d_ellipf_dm
# 
# def test_gradellipf_dphi_typical():
#     g_f1 = float(grad(finc, argnums=1)(φ0, m0))
#     g_hp = high_precison_dfdm(φ0, m0)
#     assert almosteq(g_f1, g_hp, rel_eps=1e-17)
# 
# def test_gradellipf_dphi_near_large_phi():
#     φ = jnp.pi/2 - 1e-4
#     m = -1.
#     g_f1 = float(grad(finc, argnums=1)(φ, m))
#     g_hp = high_precison_dfdm(φ, m)
#     assert almosteq(g_f1, g_hp, rel_eps=4e-17)
# 
# def test_gradellipf_dphi_phi_at_zero():
#     g_f1 = grad(finc)(0., 0.1)
#     assert g_f1 == 1.0
# 
# def test_gradellipf_dphi_m_neginf():
#     g_f1 = grad(finc)(1., -INF)
#     assert g_f1 == 0.0
# 
# def test_gradellipf_dphi_phi_zero_m_neginf():
#     g_f1 = grad(finc)(0., -INF)
#     assert g_f1 == 0.0

### Test ellipeinc
# def test_ellipe_phi_zero():
#     e1 = einc(0., m0)
#     assert e1 == 0.
# 
# def test_ellipe_m_posinf():
#     e1 = einc(φ0, INF)
#     assert isnan(e1)
# 
# def test_ellipe_m_neginf():
#     e1 = einc(φ0, -INF)
#     assert isinf(e1)
# 
# def test_ellipe_phi_zero_m_neginf():
#     e1 = einc(0., -INF)
#     assert isinf(e1)
# 
# def test_ellipe_m_zero():
#     e1 = einc(φ0, 0.)
#     assert e1 == φ0
# 
# def test_ellipe_phi_nan():
#     e1 = einc(NAN, m0)
#     assert isnan(e1)
# 
# def test_ellipe_m_nan():
#     e1 = einc(φ0, nan)
#     assert isnan(e1)
# 
# def test_ellipe_phi_nan_m_nan():
#     e1 = einc(NAN, NAN)
#     assert isnan(e1)
# 
# def test_fused_form_equality():
#     f1 = finc(φ0, m0)
#     e1 = einc(φ0, m0)
#     f2, e2 = legendre.ellip_finc_einc_fused(φ0, m0)
#     assert f1 == f2
#     assert e1 == e2

