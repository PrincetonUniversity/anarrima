from anarrima.elliptic.carlson import rf, rd
import jax
import jax.numpy as jnp

sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos

@jax.custom_jvp
def finc(φ, m):
    """Incomplete elliptic integral of the first kind """
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * rf(x, y, z)

@jax.custom_jvp
def einc(φ, m):
    """Incomplete elliptic integral of the second kind """
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    sin_cu_φ = sin_sq_φ * sinφ
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * rf(x, y, z) - m * sin_cu_φ * rd(x, y, z) / 3

@jax.jit
def f_e_fused(φ, m):
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    sin_cu_φ = sin_sq_φ * sinφ
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    ellip_f = sinφ * rf(x, y, z)
    ellip_e = ellip_f - m * sin_cu_φ * rd(x, y, z) / 3
    return ellip_f, ellip_e

# This will fail to evaluate if m==0 or m==1.
# However, since m = -4 p r / ((p-r)² + z²) in the Anarrima package geometry,
# it is strictly negative for physically meaningful values of m
@finc.defjvp
def finc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents

    finc, einc = f_e_fused(φ, m)

    primal_out = finc

    d_ellipf_dφ = 1/sqrt(1 - m*sin(φ)**2)

    # compute dF/dm
    d_L = sin(2 * φ) / (4 * (m-1) * sqrt(1-m*sin(φ)**2))
    d_F = -finc/(2 * m)
    d_E = -einc/(2 * (m-1) * m)
    d_ellipf_dm = d_L + d_F + d_E

    tangent_out = d_ellipf_dφ * φ_dot + d_ellipf_dm * m_dot
    return primal_out, tangent_out

@einc.defjvp
def einc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents
    finc, einc = f_e_fused(φ, m)
    primal_out = einc

    d_ellipe_dφ = sqrt(1 - m*sin(φ)**2)
    d_ellipe_dm = (einc - finc)/(2*m)

    tangent_out = d_ellipe_dφ * φ_dot + d_ellipe_dm * m_dot
    return primal_out, tangent_out

##############################
# Treatment of Legendre K(m)
##############################

def k(m):
    """Legendre K"""
    use_standard = m > -1e10
    standard_eval = rf(0., 1. - m, 1.)
    # backup series for large, negative m
    large_neg_series_eval = _k_series_large_negative_2_terms(m)
    return jnp.where(use_standard, standard_eval, large_neg_series_eval)

##### m very close to 1
def _km1_series_6_terms(p):
    # within 1e-16 for p < 1e-3
    # p = 1 - m
    ln2 = jnp.log(2)
    lnp = jnp.log(p)
    term_0 = (4 * ln2 - lnp) / 2
    term_1 = p * ( 4 * ln2 - lnp - 2) / 8
    term_2 = 3 * p**2 * (12 * ln2 - 3 * lnp - 7) / 128
    term_3 = 5 * p**3 * (60 * ln2 - 15 * lnp - 37) / 1536
    term_4 = 35 * p**4 * (840 * ln2 - 210 * lnp - 533) / 196608
    term_5 = 63 * p**5 * (2520 * ln2 - 630 * lnp - 1627) / 1310720
    return term_0 + (term_1 + (term_2 + (term_3 + (term_4 + term_5))))

# good for m < -1e10
def _k_series_large_negative_2_terms(m):
    w = -m # wumbo
    ln2 = jnp.log(2)
    lnw = jnp.log(w)
    root_w = jnp.sqrt(w)
    term_0 = (4 * ln2 + lnw)/(2 * root_w)
    term_1 = (2 - 4 * ln2 - lnw) / (8 * root_w * w)
    return term_0 + term_1
