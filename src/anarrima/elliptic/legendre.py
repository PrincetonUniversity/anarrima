from anarrima.elliptic.carlson import rf as elliprf
from anarrima.elliptic.carlson import _rf as _elliprf
from anarrima.elliptic.carlson import rd as elliprd
import jax
import jax.numpy as jnp

sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos

@jax.custom_jvp
def ellipfinc(φ, m):
    """Incomplete elliptic integral of the first kind """
    φ, m = jnp.broadcast_arrays(φ, m)

    # elementary tests of each variable
    phi_is_zero, m_is_zero = φ == 0., m == 0.
    phi_in_standard_range = (φ >= -jnp.pi/2) & (φ <= jnp.pi/2) & (~phi_is_zero)
    either_input_is_nan = jnp.isnan(φ) | jnp.isnan(m)
    both_notnan = ~either_input_is_nan
    phi_finite, m_finite = jnp.isfinite(φ), jnp.isfinite(m)
    m_is_neginf = jnp.isneginf(m)

    # special case: F(0, m) = 0, even for infinite m
    # special case: F(φ, -∞) = 0, for any non-nan φ
    output_is_zero = (phi_is_zero | m_is_neginf) & both_notnan

    φ_sanitized = jnp.where(phi_finite, φ, 0.)
    m_sanitized = jnp.where((phi_is_zero & m_is_neginf) | either_input_is_nan, 0., m)
    sinφ = jnp.sin(φ_sanitized)
    sin_sq_φ = jnp.square(sinφ)
    x = 1. - sin_sq_φ
    y = 1 - m_sanitized * sin_sq_φ
    z = 1.

    # out of bounds, return nan
    m_is_safe = y >= 0.
    output_is_nan = either_input_is_nan | (~m_is_safe)

    # use standard case
    both_in_standard_conditions = (m_finite & phi_in_standard_range) & both_notnan
    use_standard_case = both_in_standard_conditions & m_is_safe

    standard_eval = sinφ * elliprf(x, y, z)

    which = (0 * output_is_zero +
             1 * use_standard_case +
             2 * output_is_nan)

    ### outputs for special cases
    # zeros at m = -∞
    zeros = jnp.zeros_like(φ)
    nans = jnp.full_like(m, jnp.nan)

    return jax.lax.select_n(which,
                            zeros,
                            standard_eval,
                            nans)

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
    near_1 = m > 1 - 1e-8
    below_1 = m < 1

    is_finite = jnp.isfinite(m)

    # case_0
    # is_neginf = jnp.isneginf(m)

    # case_1
    use_large_negative = jnp.logical_and(is_finite, jnp.logical_not(above_large_negative_range))

    # case_2
    use_standard = jnp.logical_and(above_large_negative_range, jnp.logical_not(near_1))

    # case 3
    use_near_1 = jnp.logical_and(near_1, below_1)

    # case 4
    is_unity = m == 1.0

    # case 5
    is_more_than_1 = m > 1

    # case_6
    is_nan = jnp.isnan(m)

    zero_integer = jnp.zeros_like(m, dtype=jnp.int8)
    which = (zero_integer + 
             1 * use_large_negative +
             2 * use_standard +
             3 * use_near_1 +
             4 * is_unity +
             5 * jnp.logical_or(is_more_than_1, is_nan))

    ### evaluate using Carlson's Rf
    # the 0.0 at the end is a safe value for K
    sanitized_m = jnp.where(jnp.logical_and(is_finite, below_1), m, 0.)
    standard_eval = _ellipk(sanitized_m)

    # between 1 - 1e-8 and 1
    sanitized_m2 = jnp.where(jnp.logical_and(near_1, below_1), m, 0.999)
    near_1_eval = _km1_series_2_terms(1 - sanitized_m2)

    ### series for large, negative m
    # here the -1 is a safe value for the series treatment, which contains log(-m).
    sanitized_m3 = jnp.where(jnp.logical_and(is_finite, below_0), m, -1.)
    large_neg_series_eval = _k_series_large_negative_2_terms(sanitized_m3)

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
                            near_1_eval,
                            infty,
                            nans)

# good from -1e10 < m < 1
def _ellipk(m):
    return _elliprf(0., 1. - m, 1., n_loops=7)

# good for m < -1e10
def _k_series_large_negative_2_terms(m):
    w = -m # wumbo
    ln2 = jnp.log(2)
    lnw = jnp.log(w)
    root_w = jnp.sqrt(w)
    term_0 = (4 * ln2 + lnw)/(2 * root_w)
    term_1 = (2 - 4 * ln2 - lnw) / (8 * root_w * w)
    return term_0 + term_1

def _km1_series_2_terms(p):
    # within 1e-15ish for m > 1 - 1e-8, i.e. p < 1e-8
    # p = 1 - m
    ln2 = jnp.log(2)
    lnp = jnp.log(p)
    term_0 = (4 * ln2 - lnp) / 2
    term_1 = p * ( 4 * ln2 - lnp - 2) / 8
    return term_0 + term_1

# m very close to 1
def ellipkm1(p):
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

##############################
# Treatment of Legendre E(m)
##############################

def _ellipe(m):
    rf = elliprf(0., 1. - m, 1.)
    rd_term = m * elliprd(0., 1. - m, 1.) / 3
    return rf - rd_term

# Good for 0 < p < 1e-4
# p = 1 - m
def _em1_series_3_terms(p):
    ln2 = jnp.log(2)
    lnp = jnp.log(p)
    term_0 = 1
    term_1 = p * (-1 + 4*ln2 - lnp) / 4
    term_2 = p**2 * (-13 + 24*ln2 - 6*lnp) / 64
    term_3 = 3 * p**3 * (-12 + 20*ln2 - 5*lnp) / 256
    return term_0 + (term_1 + (term_2 + term_3))

# Good for m < -1e5
def _ellipe_large_negative_m_2_terms(m):
    w = -m # wumbo
    ln2 = jnp.log(2)
    lnw = jnp.log(w)
    term_0 = w
    term_1 = (1 + 4 * ln2 + lnw)/(4)
    term_2 = (3 - 8 * ln2 - 2 * lnw)/(64 * w)
    return jax.lax.rsqrt(w) * (term_0 + (term_1 + term_2))

def ellipe(m):
    """Legendre E"""
    above_large_negative_range = m > -1e5
    below_0 = m < 0
    near_1 = m > 1 - 1e-4
    below_1 = m < 1

    is_finite = jnp.isfinite(m)

    # case_0
    is_neginf = jnp.isneginf(m)

    # case_1
    use_large_negative = jnp.logical_and(is_finite, jnp.logical_not(above_large_negative_range))

    # case_2
    use_standard = jnp.logical_and(above_large_negative_range, jnp.logical_not(near_1))

    # case_3
    use_near_1 = jnp.logical_and(near_1, below_1)

    # case 4
    is_unity = m == 1.0

    # case 5
    is_more_than_1 = m > 1

    # case_6
    is_nan = jnp.isnan(m)

    zero_integer = jnp.zeros_like(m, dtype=jnp.int8)
    which = (zero_integer +
             1 * use_large_negative +
             2 * use_standard +
             3 * use_near_1 +
             4 * is_unity +
             5 * jnp.logical_or(is_more_than_1, is_nan))

    ### series for large, negative m
    # here the -1 is a safe value for the series treatment, which contains log(-m).
    sanitized_m = jnp.where(jnp.logical_and(is_finite, below_0), m, -1)
    large_neg_series_eval = _k_series_large_negative_2_terms(sanitized_m)

    ### evaluate using Carlson's Rf
    sanitized_m2 = jnp.where(jnp.logical_and(is_finite, below_1), m, 0.5)
    standard_eval = _ellipe(sanitized_m2)

    ### Series near 1
    near_1_eval = _em1_series_3_terms(1 - sanitized_m2)


    ### outputs for special cases:
    # E(-inf) == inf
    infs = jnp.full_like(m, jnp.inf)
    # E(1) == inf
    ones = jnp.ones_like(m)
    # E(x > 1) == nan
    nans = jnp.full_like(m, jnp.nan)

    return jax.lax.select_n(which,
                            infs,
                            large_neg_series_eval,
                            standard_eval,
                            near_1_eval,
                            ones,
                            nans)
