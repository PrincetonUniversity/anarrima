from anarrima.elliptic.carlson import rf as elliprf
from anarrima.elliptic.carlson import rd as elliprd
import jax
import jax.numpy as jnp

sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos

@jax.custom_jvp
def ellipfinc(φ, m):
    """Incomplete elliptic integral of the first kind """
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * elliprf(x, y, z)

@jax.custom_jvp
def ellipeinc(φ, m):
    """Incomplete elliptic integral of the second kind """
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    sin_cu_φ = sin_sq_φ * sinφ
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * elliprf(x, y, z) - m * sin_cu_φ * elliprd(x, y, z) / 3

@jax.jit
def ellip_finc_einc_fused(φ, m):
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    sin_cu_φ = sin_sq_φ * sinφ
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    finc = sinφ * elliprf(x, y, z)
    einc = finc - m * sin_cu_φ * elliprd(x, y, z) / 3
    return finc, einc

# This will fail to evaluate if m==0 or m==1.
# However, since m = -4 p r / ((p-r)² + z²) in the Anarrima package geometry,
# it is strictly negative for physically meaningful values of m
@ellipfinc.defjvp
def ellipfinc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents

    finc, einc = ellip_finc_einc_fused(φ, m)

    primal_out = finc

    d_ellipf_dφ = 1/sqrt(1 - m*sin(φ)**2)

    # compute dF/dm
    d_L = sin(2 * φ) / (4 * (m-1) * sqrt(1-m*sin(φ)**2))
    d_F = -finc/(2 * m)
    d_E = -einc/(2 * (m-1) * m)
    d_ellipf_dm = d_L + d_F + d_E

    tangent_out = d_ellipf_dφ * φ_dot + d_ellipf_dm * m_dot
    return primal_out, tangent_out

@ellipeinc.defjvp
def ellipeinc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents
    finc, einc = ellip_finc_einc_fused(φ, m)
    primal_out = einc

    d_ellipe_dφ = sqrt(1 - m*sin(φ)**2)
    d_ellipe_dm = (einc - finc)/(2*m)

    tangent_out = d_ellipe_dφ * φ_dot + d_ellipe_dm * m_dot
    return primal_out, tangent_out

##############################
# Treatment of Legendre K(m)
##############################

def ellipk(m):
    """Legendre K"""
    above_large_negative_range = m > -1e10
    below_0 = m < 0
    below_1 = m < 1

    is_finite = jnp.isfinite(m)

    # case_0
    is_neginf = jnp.isneginf(m)

    # case_1
    use_large_negative = jnp.logical_and(is_finite, jnp.logical_not(above_large_negative_range))

    # case_2
    use_standard = jnp.logical_and(above_large_negative_range, below_1)

    # case 3
    is_unity = m == 1.0

    # case 4
    is_more_than_1 = m > 1

    # case_5
    is_nan = jnp.isnan(m)

    zero_integer = jnp.zeros_like(m, dtype=jnp.int8)
    which = (zero_integer + 
             1 * use_large_negative +
             2 * use_standard +
             3 * is_unity +
             4 * jnp.logical_or(is_more_than_1, is_nan))

    ### evaluate using Carlson's Rf
    # the 0.0 at the end is a safe value for K
    sanitized_m = jnp.where(jnp.logical_and(is_finite, below_1), m, 0.)
    standard_eval = _ellipk(sanitized_m)

    ### series for large, negative m
    # here the -1 is a safe value for the series treatment, which contains log(-m).
    sanitized_m2 = jnp.where(jnp.logical_and(is_finite, below_0), m, -1.)
    large_neg_series_eval = _k_series_large_negative_2_terms(sanitized_m2)

    ### outputs for special cases:
    # K(-inf) == 0
    zero = jnp.zeros_like(m)
    # K(1) == inf
    infty = jnp.full_like(m, jnp.inf)
    # K(x > 1) == nan
    nans = jnp.full_like(m, jnp.nan)

    return jax.lax.select_n(which,
                            zero,
                            large_neg_series_eval,
                            standard_eval,
                            infty,
                            nans)

# good from -1e10 < m < 1
def _ellipk(m):
    return elliprf(0., 1. - m, 1.)

# good for m < -1e10
def _k_series_large_negative_2_terms(m):
    w = -m # wumbo
    ln2 = jnp.log(2)
    lnw = jnp.log(w)
    root_w = jnp.sqrt(w)
    term_0 = (4 * ln2 + lnw)/(2 * root_w)
    term_1 = (2 - 4 * ln2 - lnw) / (8 * root_w * w)
    return term_0 + term_1

# m very close to 1
# this is not used because with the current number of loop
# evaluations in the rf function it is good enough, even down 
# to 1e-16 ish. So even doing  p = 1.0 - m loses precision enough
# to not make it worth it.
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

