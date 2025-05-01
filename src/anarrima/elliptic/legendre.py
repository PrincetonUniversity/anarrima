from anarrima.elliptic.carlson import rf as elliprf
from anarrima.elliptic.carlson import _rf as _elliprf
from anarrima.elliptic.carlson import rd as elliprd
import jax
import jax.numpy as jnp

sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos

def _xyz_incomplete(φ, m):
    """Common arguments to rf and rd"""
    x = jnp.square(cos(φ))
    y = 1 - m * jnp.square(sin(φ))
    z = 1.
    return x, y, z

def _ellipfinc(φ, m):
    """
    Defined for φ in [-π/2, π/2] and real m s.t. y>=0.
    """
    sinφ = jnp.sin(φ)
    x, y, z = _xyz_incomplete(φ, m)
    return sinφ * elliprf(x, y, z)

def _ellipeinc(φ, m):
    """
    Defined for φ in [-π/2, π/2] and real m s.t. y>=0.
    """
    sinφ = sin(φ)
    sin_cu_φ = sinφ * jnp.square(sinφ)
    x, y, z = _xyz_incomplete(φ, m)
    rf = elliprf(x, y, z)
    rd = elliprd(x, y, z)
    return sinφ * rf - m * sin_cu_φ * rd / 3

def ellipfinc(φ, m):
    """Incomplete elliptic integral of the first kind

    Defined for φ in [-π/2, π/2] and real m.
    """
    φ, m = jnp.broadcast_arrays(φ, m)

    # elementary tests of each variable
    phi_in_standard_range = (φ >= -jnp.pi/2) & (φ <= jnp.pi/2)
    either_is_nan = jnp.isnan(φ) | jnp.isnan(m)
    phi_finite, m_finite = jnp.isfinite(φ), jnp.isfinite(m)
    m_is_neginf = jnp.isneginf(m)

    m_sanitized = jnp.where(m_is_neginf | either_is_nan, 0., m)
    sinφ = jnp.sin(φ)
    y = 1 - m_sanitized * jnp.square(sinφ)

    # out of bounds, return nan
    m_is_safe = y >= 0.
    output_is_nan = either_is_nan | ~m_is_safe

    # use standard case
    use_standard_case = m_finite & phi_in_standard_range & m_is_safe

    standard_eval = _ellipfinc(φ, m_sanitized)

    ### outputs for special cases
    zeros = jnp.zeros_like(φ)
    result = jnp.where(use_standard_case, standard_eval, zeros)
    result = jnp.where(output_is_nan, jnp.nan, result)

    return result

def ellipeinc(φ, m):
    """Incomplete elliptic integral of the second kind.
    Defined for φ in [-π/2, π/2] and real m.
    """
    φ, m = jnp.broadcast_arrays(φ, m)

    phi_is_zero = φ == 0.
    phi_in_standard_range = (φ >= -jnp.pi/2) & (φ <= jnp.pi/2) & ~phi_is_zero
    m_is_neginf = jnp.isneginf(m)
    phi_finite, m_finite = jnp.isfinite(φ), jnp.isfinite(m)
    either_is_nan = jnp.isnan(φ) | jnp.isnan(m)

    y = 1. - m * jnp.square(sin(φ))

    m_is_safe = y >= 0.
    output_is_nan = either_is_nan | (~m_is_safe & ~m_is_neginf)
    output_is_inf = phi_finite & m_is_neginf

    use_standard_case = m_finite & phi_in_standard_range & m_is_safe

    standard_eval = _ellipeinc(φ, m)
    zeros = jnp.zeros_like(φ)
    result = jnp.where(use_standard_case, standard_eval, zeros)
    result = jnp.where(output_is_inf, jnp.inf, result)
    result = jnp.where(output_is_nan, jnp.nan, result)
    return result

def ellip_finc_einc_fused(φ, m):
    sinφ = sin(φ)
    sin_cu_φ = sinφ * jnp.square(sinφ)
    x, y, z = _xyz_incomplete(φ, m)
    finc = sinφ * elliprf(x, y, z)
    einc = finc - m * sin_cu_φ * elliprd(x, y, z) / 3
    return finc, einc

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
