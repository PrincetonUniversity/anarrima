import jax.numpy as jnp
from anarrima.elliptic.legendre import ellipeinc, ellipfinc

sin = jnp.sin
cos = jnp.cos
sqrt = jnp.sqrt


def amplitude_and_parameter(p, z, r, φ):
    """Common inputs to the elliptic integrals"""
    varφ = φ / 2
    m = -4 * p * r / ((p - r) ** 2 + z**2)
    return varφ, m


def q_term(p, z, r):
    """A common term in the reduced integrals"""
    return p**4 - 2 * p**2 * (r**2 - z**2) + (r**2 + z**2) ** 2


###############################################
# Key for function names
#
# [gh]_[HV][IABc][a][{LEKF}_term]
#  |    |   |     |   |
#  |    |   |     |   L - polynomial term
#  |    |   |     |   E - elliptic E coefficient
#  |    |   |     |   K - elliptic K coefficient
#  |    |   |     |   F - elliptic F coefficient
#  |    |   |     |
#  |    |   |     a - angled field
#  |    |   |
#  |    |   I - Isotropic distribution
#  |    |   A - "A-mode" proportional to sin²θ
#  |    |   B - "B-mode" proportional to 1 + cos²θ
#  |    |   c - cos²θ, used to make the B-mode
#  |    |
#  |    H - horizontal-normal component
#  |    V - vertical-normal component
#  |
#  g - reduced integrals
#  h - individual terms which sum to g; also has a


###############################################

##################################
# Isotropic horizontal same height
##################################


# used for testing
def g_HI_same_height(*, p, r, φ):
    h_HI_same_height_L = (4 * p**2 * sin(φ)) / (
        (p**2 - r**2) * sqrt(p**2 + r**2 - 2 * p * r * cos(φ))
    )

    varφ, m = amplitude_and_parameter(p, 0, r, φ)
    h_HI_same_height_E = 2 * p * ellipeinc(varφ, m) / (r * (p + r))
    h_HI_same_height_K = -2 * p * ellipfinc(varφ, m) / (r * (p - r))
    return h_HI_same_height_L + h_HI_same_height_E + h_HI_same_height_K


# fmt: off

#########################
# Isotropic horizontal
#########################

def h_HIL_term(p, z, r, φ):
    """Horizontal, isotropic: leading term"""
    q = q_term(p, z, r)
    numer = 4 * p**2 * (p**2 - r**2 + z**2) * sin(φ)
    denom  = q * sqrt(p**2 + r**2 + z**2 - 2 * p * r * cos(φ))
    return numer/denom

def h_HIE_term(p, z, r, φ):
    """Horizontal, isotropic: E term"""
    q = q_term(p, z, r)
    return 2 * p * sqrt((p-r)**2 + z**2) * (p**2 - r**2 + z**2) / (q * r)

def h_HIK_term(p, z, r, φ):
    """Horizontal, isotropic: K term"""
    return (-2 * p ) / (r * sqrt((p - r)**2 + z**2))

def g_HI(p, z, r, φ):
    """Horizontal, isotropic"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    h_HIL = h_HIL_term(p, z, r, φ)
    h_HIE = h_HIE_term(p, z, r, φ) * ellipeinc(varφ, m)
    h_HIK = h_HIK_term(p, z, r, φ) * ellipfinc(varφ, m)
    return h_HIL + h_HIE + h_HIK

#########################
# Isotropic vertical
#########################

def g_VI(p, z, r, φ):
    """Vertical, isotropic"""
    # L term
    p2, r2, z2 = p**2, r**2, z**2

    numer = 8 * p2 * r * z * sin(φ)
    denom = (((p2 + r2 + z2)**2 - 4*p2*r2) *
                   sqrt(p2 + r2 + z2 - 2*p*r*cos(φ)))
    h_VIL = numer / denom

    # E term
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    numer = 4 * p * z * ellipeinc(varφ, m)
    denom = sqrt((p - r)**2 + z2) * ((p + r)**2 + z2)
    h_VIE = numer / denom

    return h_VIL + h_VIE


####################################
# Horizontal, A-mode, toroidal field
####################################

def h_HAL(p, z, r, φ):
    q = q_term(p, z, r)

    sinφ = sin(φ)
    cosφ = cos(φ)
    z2 = z**2
    p2 = p**2

    # L term
    numerator1 = -2 * sinφ * (
        -4*p**6 + r**6 - 6*p**4*z2 + 2*z**6 + 2*r**4*(p2 + 2*z2) +
        r**2*(p2 + z2)*(p2 + 5*z2) -
        p*r*(-7*p**4 + r**4 - 2*p2*z2 + 5*z**4 + 6*r**2*(p2 + z2))*cosφ
    )
    denominator1 = 3 * q * (p2 + r**2 + z2 - 2*p*r*cos(φ))**(3/2)
    h_HAL = numerator1 / denominator1

    return h_HAL

def h_HAF(p, z, r, φ):
    varφ, m = amplitude_and_parameter(p, z, r, φ)

    z2 = z**2
    p2 = p**2

    return 2*(-p2 + r**2 + 2*z2)*ellipfinc(varφ, m) / \
            (3*p*r*sqrt((p - r)**2 + z2))

def h_HAE(p, z, r, φ):
    """Horizontal, A-mode, E term"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    z2 = z**2
    p2 = p**2

    h_HAE = -2*sqrt((p - r)**2 + z2)*(-p**4 + r**4 + (p2 + 3*r**2)*z2 + 2*z**4)* \
            ellipeinc(varφ, m)/(3*p*q*r)

    return h_HAE

def g_HA(p, z, r, φ):
    """Horizontal, A-mode"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    sinφ = sin(φ)
    cosφ = cos(φ)
    z2 = z**2
    p2 = p**2

    # L term
    numerator1 = -2 * sinφ * (
        -4*p**6 + r**6 - 6*p**4*z2 + 2*z**6 + 2*r**4*(p2 + 2*z2) +
        r**2*(p2 + z2)*(p2 + 5*z2) -
        p*r*(-7*p**4 + r**4 - 2*p2*z2 + 5*z**4 + 6*r**2*(p2 + z2))*cosφ
    )
    denominator1 = 3 * q * (p2 + r**2 + z2 - 2*p*r*cos(φ))**(3/2)
    h_HAL = numerator1 / denominator1

    h_HAF = 2*(-p2 + r**2 + 2*z2)*ellipfinc(varφ, m) / \
            (3*p*r*sqrt((p - r)**2 + z2))

    h_HAE = -2*sqrt((p - r)**2 + z2)*(-p**4 + r**4 + (p2 + 3*r**2)*z2 + 2*z**4)* \
            ellipeinc(varφ, m)/(3*p*q*r)

    return h_HAL + h_HAF + h_HAE

####################################
# Horizontal, cosine, toroidal field
####################################

def g_Hc(p, z, r, φ):
    """Horizontal, cosine"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    cosφ, sinφ = cos(φ), sin(φ)
    p2, r2, z2 = p**2, r**2, z**2
    p4, r4, z4 = p2**2, r2**2, z2**2
    p6 = p**6

    # First term (h_HcL)
    frac = 2 * sinφ / (3 * q * (p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer_L = (
        2*p6 + (r2 + z2)**2*(r2 + 2*z2) + p4*(r2 + 6*z2) +
        p2*(-4*r4 + 6*r2*z2 + 6*z4) -
        p*r*(5*p4 - 6*p2*r2 + r4 + 2*(5*p2 + 3*r2)*z2 + 5*z4)*cosφ
    )
    h_HcL = frac * numer_L

    # Second term (hHcF) - elliptic integral of first kind
    h_HcF = -2*(2*p2 + r2 + 2*z2) / (3*p*r*sqrt((p - r)**2 + z2))

    # Third term (hHcE) - elliptic integral of second kind
    numer_E = 2*(2*p4 - 3*p2*r2 + r4 + (4*p2 + 3*r2)*z2 + 2*z4)
    denom_E = (3*p*r*sqrt((p - r)**2 + z2)*((p + r)**2 + z2))
    h_HcE = numer_E / denom_E

    return h_HcL + h_HcF*ellipfinc(varφ, m) + h_HcE*ellipeinc(varφ, m)

####################################
# Horizontal, B-mode, toroidal field
####################################

def g_HB(p, z, r, φ):
    """Horizontal, B-mode"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    cosφ, sinφ = cos(φ), sin(φ)
    p2, r2, z2 = p**2, r**2, z**2
    p4, r4, z4 = p2**2, r2**2, z2**2
    p6 = p**6

    # First term (h_HcL)
    frac = sinφ / (2 * q * (p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    term_1 = 4 * p6 + (r2 + z2)**2 * (r2 + 2 * z2)   
    term_2 = p4 * (r2 + 10 * z2) + p2 * (-6 * r4 + 6 * r2 * z2 + 8 * z4)
    term_3 = - p * r * (9 * p4 + r4 + 6 * r2 * z2 + 5 * z4 + 2 * p2 * (-5 * r2 + 7 * z2))
    numer_L = term_1 + term_2 + term_3 * cosφ
    h_HcL = frac * numer_L

    # Second term (hHcF) - elliptic integral of first kind
    h_HcF = -(3*p2 + r2 + 2*z2) * ellipfinc(varφ, m) / (2*p*r*sqrt((p - r)**2 + z2))

    # Third term (hHcE) - elliptic integral of second kind
    frac_E = sqrt((p - r)**2 + z2) * ellipeinc(varφ, m) / (2 * p * q * r)
    numer_E = (3 * p4 + r4 + 3 * r2 * z2 + 2 * z4 + p2 * (-4 * r2 + 5 * z2))
    h_HcE = frac_E * numer_E

    return h_HcL + h_HcF + h_HcE

####################################
# Horizontal, A-mode, angled field
####################################

def g_HAa(p, z, r, α, β, φ):
    """Horizontal, A-mode, angled field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    # Precompute powers
    p2, r2, z2 = p**2, r**2, z**2
    r3 = r2*r
    p4, r4, z4 = p2**2, r2**2, z2**2
    p6, r6, z6 = p4*p2, r4*r2, z4*z2
    p8, r8, z8 = p4**2, r4**2, z4**2
    p10 = p8*p2

    # Precompute trig functions
    cosφ, sinφ = cos(φ), sin(φ)
    cos_2α, sin_2α = cos(2*α), sin(2*α)
    sin_β_sq = sin(β)**2

    # Leading term (h_HAaL)
    term_1 = 2*q*(4*p6 - p4*(r2 - 6*z2) - (r2 + z2)**2*(r2 + 2*z2) -
                 2*p2*(r4 + 3*r2*z2) +
                 p*r*(-7*p4 + 6*p2*r2 + r4 - 2*(p2 - 3*r2)*z2 + 5*z4)*cosφ)

    term_2 = (3*q*(((-p2*r + r3)**2 + 2*(p2 + r2)*(p2 + 2*r2)*z2 +
                   (4*p2 + 5*r2)*z4 + 2*z6) -
                  p*r*(((p2 - r2)**2 + 6*(p2 + r2)*z2 + 5*z4)*cosφ)))

    term_3 = (-4*p10 + p8*(9*r2 - 6*z2) + (r2 + z2)**4*(r2 + 2*z2) +
               4*p2*z2*(r2 + z2)**2*(-7*r2 + 3*z2) -
               4*p6*(r4 - 7*r2*z2 - 2*z4) +
               p4*(-2*r6 + 6*r2*z4 + 20*z6) +
               p*r*((p2 - r2)**3*(7*p2 + r2) +
                   8*(p6 - 7*p4*r2 + 7*p2*r4 - r6)*z2 -
                   2*(5*p4 - 22*p2*r2 + 9*r4)*z4 -
                   16*(p2 + r2)*z6 - 5*z8)*cosφ)

    term_4 = 2*p*z*(5*p8 - 4*p6*(r2 - 4*z2) + (r2 + z2)**3*(5*r2 + z2) -
                   2*p4*(r4 + 8*r2*z2 - 9*z4) -
                   4*p2*(r6 + 4*r4*z2 + r2*z4 - 2*z6) -
                   4*p*r*(2*(p2 - r2)**2*(p2 + r2) +
                         5*(p2 - r2)**2*z2 + 4*(p2 + r2)*z4 +
                         z6)*cosφ)

    frac = sinφ/ (3*q**2*(p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer = (term_1 + (term_2 + term_3 * cos_2α + term_4 * sin_2α)*sin_β_sq)
    h_HAaL = frac * numer

    # Second term (h_HAaF) - elliptic integral of first kind
    frac = 1/(3*p*q*r*sqrt((p - r)**2 + z2))
    numer = (2*q*(-p2 + r2 + 2*z2) +
        (-3*q*(p2 + r2 + 2*z2) +
         (p6 - (r2 + z2)**2*(r2 + 2*z2) - p4*(3*r2 + 2*z2) +
          p2*(3*r4 + 6*r2*z2 - 5*z4))*cos_2α -
         2*p*z*(2*p4 + 2*r4 + 3*r2*z2 + z4 + p2*(-4*r2 + 3*z2))*sin_2α)*sin_β_sq
    )
    h_HAaF  = frac * numer * ellipfinc(varφ, m)

    # Third term (h_HAaE) - elliptic integral of second kind
    frac = sqrt((p - r)**2 + z2) / (3 * p* q**2 * r)
    term_1 = -2*q*(-p4 + r4 + (p2 + 3*r2)*z2 + 2*z4)
    term_2 = 3*q*((p2 - r2)**2 + 3*(p2 + r2)*z2 + 2*z4)
    term_3 = (-p8 + p6*(2*r2 + z2) + (r2 + z2)**3*(r2 + 2*z2) -
          p2*(r2 + z2)*(2*r4 + 15*r2*z2 - 7*z4) +
          p4*(11*r2*z2 + 7*z4))
    term_4 = 2*p*z*(2*(p2 - r2)**2 * (p2 + r2) + 5*(p2 - r2)**2*z2 + 4*(p2 + r2)*z4 + z6)

    numer = term_1 + (term_2 + term_3 * cos_2α + term_4 * sin_2α)*sin_β_sq
    h_HAaE = frac * numer * ellipeinc(varφ, m)
    return h_HAaL + h_HAaF + h_HAaE

####################################
# Horizontal, cosine, angled field
####################################

def g_Hca(p, z, r, α, β, φ):
    """Horizontal, cosine, angled field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    # Precompute powers
    p2, r2, z2 = p**2, r**2, z**2
    p3, r3 = p2*p, r2*r
    p4, r4, z4 = p2**2, r2**2, z2**2
    p6, r6, z6 = p4*p2, r4*r2, z4*z2
    p8, r8, z8 = p4**2, r4**2, z4**2
    p10 = p8*p2

    # Precompute trig functions
    cosφ, sinφ = cos(φ), sin(φ)
    cos_2α, sin_2α = cos(2*α), sin(2*α)
    sin_β_sq = sin(β)**2

    # Leading term (h_HcaL)
    term_1 = q*(-8*p2*r2*(p2 + r2 + z2) + 4*(p2 + r2 + z2)**3 +
                24*p3*r3*cosφ - 10*p*r*(p2 + r2 + z2)**2*cosφ -
                2*r2*(4*p2*r2 + (p2 + r2 + z2)**2 -
                      4*p*r*(p2 + r2 + z2)*cosφ))

    term_2 = 3*q*(-p4*(r2 + 2*z2) - (r2 + z2)**2*(r2 + 2*z2) +
                  2*p2*(r4 - 3*r2*z2 - 2*z4) +
                  p*r*((p2 - r2)**2 + 6*(p2 + r2)*z2 + 5*z4)*cosφ)

    term_3 = (4*p10 + 4*p2*z2*(7*r2 - 3*z2)*(r2 + z2)**2 - (r2 + z2)**4*(r2 + 2*z2) +
              p8*(-9*r2 + 6*z2) + 4*p6*(r4 - 7*r2*z2 - 2*z4) +
              2*p4*(r6 - 3*r2*z4 - 10*z6) +
              p*r*(-7*p8 + 4*p6*(5*r2 - 2*z2) + (r2 + z2)**3*(r2 + 5*z2) +
                   4*p2*(r2 + z2)*(r4 - 15*r2*z2 + 4*z4) +
                   2*p4*(-9*r4 + 28*r2*z2 + 5*z4))*cosφ)

    term_4 = -2*p*z*(5*p8 - 4*p6*(r2 - 4*z2) + (r2 + z2)**3*(5*r2 + z2) -
                     2*p4*(r4 + 8*r2*z2 - 9*z4) -
                     4*p2*(r6 + 4*r4*z2 + r2*z4 - 2*z6) -
                     4*p*r*(2*(p2 - r2)**2*(p2 + r2) +
                           5*(p2 - r2)**2*z2 + 4*(p2 + r2)*z4 + z6)*cosφ)

    frac = sinφ/(3*q**2*(p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_HcaL = frac * numer

    # Second term (h_HcaF) - elliptic integral of first kind
    frac = -1/(3*p*q*r*sqrt((p - r)**2 + z2))
    term_1 = 2*q*(2*p2 + r2 + 2*z2)
    term_2 = -3*q*(p2 + r2 + 2*z2)
    term_3 = ((p2 - r2)**3 - 2*(p4 - 3*p2*r2 + 2*r4)*z2 - 5*(p2 + r2)*z4 - 2*z6)
    term_4 = 2*p*z*(-2*(p2 - r2)**2 - 3*(p2 + r2)*z2 - z4)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_HcaF = frac * numer * ellipfinc(varφ, m)

    # Third term (h_HcaE) - elliptic integral of second kind
    frac = sqrt((p - r)**2 + z2)/(3*p*r*q**2)
    term_1 = 2*q*(2*p4 + r4 + 3*r2*z2 + 2*z4 + p2*(-3*r2 + 4*z2))
    term_2 = -3*q*((p2 - r2)**2 + 3*(p2 + r2)*z2 + 2*z4)
    term_3 = (p8 - p6*(2*r2 + z2) - (r2 + z2)**3*(r2 + 2*z2) +
              p2*(r2 + z2)*(2*r4 + 15*r2*z2 - 7*z4) - p4*(11*r2*z2 + 7*z4))
    term_4 = 2*p*z*(-2*(p2 - r2)**2*(p2 + r2) - 5*(p2 - r2)**2*z2 -
                    4*(p2 + r2)*z4 - z6)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_HcaE = frac * numer * ellipeinc(varφ, m)

    return h_HcaL + h_HcaF + h_HcaE

####################################
# Horizontal, B-mode, angled field
####################################

def g_HBa(p, z, r, α, β, φ):
    """Horizontal, B-mode, angled field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    # Precompute powers
    p2, r2, z2 = p**2, r**2, z**2
    p4, r4, z4 = p2**2, r2**2, z2**2
    p6, r6, z6 = p4*p2, r4*r2, z4*z2
    p8 = p4**2
    p10 = p8*p2

    # Precompute trig functions
    cosφ, sinφ = cos(φ), sin(φ)
    cos_2α, sin_2α = cos(2*α), sin(2*α)
    sin_β_sq = sin(β)**2

    # Leading term (h_HBaL)
    term_1 = 2*q*(4*p6 + (r2 + z2)**2*(r2 + 2*z2) +
                  p4*(r2 + 10*z2) + p2*(-6*r4 + 6*r2*z2 + 8*z4) -
                  p*r*(9*p4 + r4 + 6*r2*z2 + 5*z4 +
                       2*p2*(-5*r2 + 7*z2))*cosφ)

    term_2 = 3*q*(-p4*(r2 + 2*z2) - (r2 + z2)**2*(r2 + 2*z2) +
                  2*p2*(r4 - 3*r2*z2 - 2*z4) +
                  p*r*((p2 - r2)**2 + 6*(p2 + r2)*z2 + 5*z4)*cosφ)

    term_3 = (4*p10 + 4*p2*z2*(7*r2 - 3*z2)*(r2 + z2)**2 - (r2 + z2)**4*(r2 + 2*z2) +
              p8*(-9*r2 + 6*z2) + 4*p6*(r4 - 7*r2*z2 - 2*z4) +
              2*p4*(r6 - 3*r2*z4 - 10*z6) +
              p*r*(-7*p8 + 4*p6*(5*r2 - 2*z2) + (r2 + z2)**3*(r2 + 5*z2) +
                   4*p2*(r2 + z2)*(r4 - 15*r2*z2 + 4*z4) +
                   2*p4*(-9*r4 + 28*r2*z2 + 5*z4))*cosφ)

    term_4 = -2*p*z*(5*p8 - 4*p6*(r2 - 4*z2) + (r2 + z2)**3*(5*r2 + z2) -
                     2*p4*(r4 + 8*r2*z2 - 9*z4) -
                     4*p2*(r6 + 4*r4*z2 + r2*z4 - 2*z6) -
                     4*p*r*(2*(p2 - r2)**2*(p2 + r2) +
                           5*(p2 - r2)**2*z2 + 4*(p2 + r2)*z4 + z6)*cosφ)

    frac = sinφ/(4*q**2*(p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_HBaL = frac * numer

    # Second term (h_HBaF) - elliptic integral of first kind
    frac = 1/(4*p*q*r*sqrt((p - r)**2+ z2))
    term_1 = -2*q*(3*p2 + r2 + 2*z2)
    term_2 = 3*q*(p2 + r2 + 2*z2)
    term_3 = (-(p2 - r2)**3 + 2*(p4 - 3*p2*r2 + 2*r4)*z2 + 5*(p2 + r2)*z4 + 2*z6)
    term_4 = 2*p*z*(2*(p2 - r2)**2 + 3*(p2 + r2)*z2 + z4)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_HBaF = frac * numer * ellipfinc(varφ, m)

    # Third term (h_HBaE) - elliptic integral of second kind
    frac = sqrt((p - r)**2 + z2)/(4*p*q**2*r)
    term_1 = 2*q*(3*p4 + r4 + 3*r2*z2 + 2*z4 + p2*(-4*r2 + 5*z2))
    term_2 = -3*q*((p2 - r2)**2 + 3*(p2 + r2)*z2 + 2*z4)
    term_3 = (p8 - p6*(2*r2 + z2) - (r2 + z2)**3*(r2 + 2*z2) +
              p2*(r2 + z2)*(2*r4 + 15*r2*z2 - 7*z4) - p4*(11*r2*z2 + 7*z4))
    term_4 = 2*p*z*(-2*(p2 - r2)**2*(p2 + r2) - 5*(p2 - r2)**2*z2 -
                    4*(p2 + r2)*z4 - z6)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_HBaE = frac * numer * ellipeinc(varφ, m)

    return h_HBaL + h_HBaF + h_HBaE

####################################
# Vertical, A-mode, toroidal field
####################################

def g_VA(p, z, r, φ):
    """Vertical, A-mode, toroidal field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)
    p2, r2, z2 = p**2, r**2, z**2
    p4 = p2**2

    cosφ = cos(φ)
    sinφ = sin(φ)

    term_1 = -11 * p4 + (r2 + z2)**2 - 2 * p2 * (3 * r2 + 5 * z2)
    term_2 = 4 * p * r * (5 * p2 - r2 - z2) * cosφ
    numer = - 2 * r * z * (term_1 + term_2) * sinφ
    denom = 3 * q * (p2 + r2 + z2 - 2 * p * r * cosφ)**(3/2)
    h_VAL = numer / denom

    h_VAF = 2 * z * ellipfinc(varφ, m) / \
        (3 * p * sqrt((p - r)**2 + z2))
    h_VAE = 2 * z * (5 * p2 - r2 - z2) * sqrt(p2 - 2 * p * r + r2 + z2) * \
        ellipeinc(varφ, m) / (3*p*q)

    return h_VAL + h_VAF + h_VAE

####################################
# Vertical, cosine, toroidal field
####################################

def g_Vc(p, z, r, φ):
    """Vertical, cosine, toroidal field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    p2, r2, z2 = p**2, r**2, z**2
    p4 = p2**2

    cosφ = cos(φ)
    sinφ = sin(φ)
    term_1 = p4 + (r2 + z2)**2 + 2 * p2 * (3 * r2 + z2)
    term_2 = - 4 * p * r * (p2 + r2 + z2) * cosφ
    numer = 2 * r * z * (term_1 + term_2) * sinφ
    denom = 3 * q * (p2 + r2 + z2 - 2 * p * r * cosφ)**(3/2)
    h_VcL = numer / denom

    h_VcF = - 2 * z * ellipfinc(varφ, m) / \
        (3 * p * sqrt((p-r)**2 + z2))
    h_VcE = 2 * z * (p2 + r2 + z2) * sqrt((p - r)**2 + z2) * \
        ellipeinc(varφ, m) / (3*p*q)

    return h_VcL + h_VcF + h_VcE

####################################
# Vertical, B-mode, toroidal field
####################################

def g_VB(p, z, r, φ):
    """Vertical, B-mode, toroidal field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    p2, r2, z2 = p**2, r**2, z**2
    p4 = p2**2

    cosφ = cos(φ)
    sinφ = sin(φ)

    term_1 = 5 * p4 + (r2 + z2)**2 + 2 * p2 * (5 * r2 + 3 * z2)
    term_2 = -4 * p * r * (3 * p2 + r2 + z2) * cosφ
    numer = r * z * (term_1 + term_2) * sinφ
    denom = 2 * q * (p2 + r2 + z2 - 2 * p * r * cosφ)**(3/2)
    h_VBL = numer / denom

    h_VBF = -z * ellipfinc(varφ, m) / (2 * p * sqrt((p - r)**2 + z2))

    h_VBE = z * (3 * p2 + r2 + z2) * sqrt((p - r)**2 + z2) * \
            ellipeinc(varφ, m) / (2 * p * q)

    return h_VBL + h_VBF + h_VBE

####################################
# Vertical, A-mode, angled field
####################################

def g_VAa(p, z, r, α, β, φ):
    """Vertical, A-mode, angled field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    # Precompute powers
    p2 = p**2
    p4 = p2**2
    p6 = p4*p2
    p8 = p4**2
    r2 = r**2
    r4 = r2**2
    z2 = z**2
    z4 = z2**2

    # Precompute trig functions
    cosφ = cos(φ)
    sinφ = sin(φ)
    cos_2α, sin_2α = cos(2*α), sin(2*α)
    sin_β_sq = sin(β)**2

    # Leading term (hVAaL)
    term_1 = -2*q*r*z*(-11*p4 + (r2 + z2)**2 - 2*p2*(3*r2 + 5*z2) +
                       4*p*r*(5*p2 - r2 - z2)*cosφ)

    term_2 = 3*q*r*z*(-3*p4 + 2*p2*(r - z)*(r + z) + (r2 + z2)**2 +
                      4*p*r*(p2 - r2 - z2)*cosφ)

    term_3 = r*z*(-11*p8 - 8*p2*(r2 - 4*z2)*(r2 + z2)**2 + (r2 + z2)**4 +
                  8*p6*(2*r2 + z2) + 2*p4*(r4 + 18*r2*z2 + 25*z4) +
                  4*p*r*(5*p6 + p2*(7*r2 - 13*z2)*(r2 + z2) - (r2 + z2)**3 -
                        p4*(11*r2 + 7*z2))*cosφ)

    term_4 = -8*p*r*z2*(-4*p6 + p4*(r2 - 7*z2) + (r2 + z2)**3 +
                        2*p2*(r4 - z4) +
                        p*r*(7*p4 + 6*p2*(-r2 + z2) - (r2 + z2)**2)*cosφ)

    frac = sinφ/(3*q**2*(p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VAaL = frac * numer

    # Second term (hVAaF) - elliptic integral of first kind
    frac = 1/(3*p*q*sqrt((p - r)**2 + z2))
    term_1 = 2*q
    term_2 = -3*q
    term_3 = (-p4 + 2*p2*(r2 - 3*z2) - (r2 + z2)**2)
    term_4 = 2*p*z*(-p2 + r2 + z2)
    numer = z*(term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq)
    h_VAaF = frac * numer * ellipfinc(varφ, m)

    # Third term (hVAaE) - elliptic integral of second kind
    frac = z*sqrt(p2 - 2*p*r + r2 + z2)/(3*p*q**2)
    term_1 = 2*q*(5*p2 - r2 - z2)
    term_2 = 3*q*(-p2 + r2 + z2)
    term_3 = (-5*p6 + (r2 + z2)**3 + p4*(11*r2 + 7*z2) +
              p2*(-7*r4 + 6*r2*z2 + 13*z4))
    term_4 = 2*p*z*(7*p4 - 6*p2*(r2 - z2) - (r2 + z2)**2)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VAaE = frac * numer * ellipeinc(varφ, m)

    return h_VAaL + h_VAaF + h_VAaE

def g_Vca(p, z, r, α, β, φ):
    """Vertical, cosine, angled field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    # Precompute powers
    p2 = p**2
    p4 = p2**2
    p6 = p4*p2
    p8 = p4**2
    r2 = r**2
    r4 = r2**2
    z2 = z**2
    z4 = z2**2

    # Precompute trig functions
    cosφ, sinφ = cos(φ), sin(φ)
    cos_2α, sin_2α = cos(2*α), sin(2*α)
    sin_β_sq = sin(β)**2

    # Leading term (hVCaL)
    term_1 = 2*q*r*z*(p4 + (r2 + z2)**2 + 2*p2*(3*r2 + z2) -
                      4*p*r*(p2 + r2 + z2)*cosφ)

    term_2 = -3*q*r*z*(-3*p4 + 2*p2*(r - z)*(r + z) + (r2 + z2)**2 +
                       4*p*r*(p2 - r2 - z2)*cosφ)

    term_3 = -r*z*(-11*p8 - 8*p2*(r2 - 4*z2)*(r2 + z2)**2 + (r2 + z2)**4 +
                   8*p6*(2*r2 + z2) + 2*p4*(r4 + 18*r2*z2 + 25*z4) +
                   4*p*r*(5*p6 + p2*(7*r2 - 13*z2)*(r2 + z2) - (r2 + z2)**3 -
                         p4*(11*r2 + 7*z2))*cosφ)

    term_4 = 8*p*r*z2*(-4*p6 + p4*(r2 - 7*z2) + (r2 + z2)**3 +
                       2*p2*(r4 - z4) +
                       p*r*(7*p4 + 6*p2*(-r2 + z2) - (r2 + z2)**2)*cosφ)

    frac = sinφ/(3*q**2*(p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VCaL = frac * numer

    # Second term (hVCaF) - elliptic integral of first kind
    frac = z/(3*p*sqrt((p - r)**2 + z2)*q)
    term_1 = -2*q
    term_2 = 3*q
    term_3 = (p4 - 2*p2*(r2 - 3*z2) + (r2 + z2)**2)
    term_4 = 2*p*z*(p2 - r2 - z2)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VCaF = frac * numer * ellipfinc(varφ, m)

    # Third term (hVCaE) - elliptic integral of second kind
    frac = z*sqrt((p - r)**2 + z2)/(3*p*q**2)
    term_1 = 2*q*(p2 + r2 + z2)
    term_2 = 3*q*(p2 - r2 - z2)
    term_3 = (5*p6 + p2*(7*r2 - 13*z2)*(r2 + z2) - (r2 + z2)**3 -
              p4*(11*r2 + 7*z2))
    term_4 = 2*p*z*(-7*p4 + 6*p2*(r - z)*(r + z) + (r2 + z2)**2)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VCaE = frac * numer * ellipeinc(varφ, m)

    return h_VCaL + h_VCaF + h_VCaE


def g_VBa(p, z, r, α, β, φ):
    """Vertical, B-mode, angled field"""
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    q = q_term(p, z, r)

    # Precompute powers
    p2 = p**2
    p4 = p2**2
    p6 = p4*p2
    p8 = p4**2
    r2 = r**2
    r4 = r2**2
    z2 = z**2
    z4 = z2**2

    # Precompute trig functions
    cosφ, sinφ = cos(φ), sin(φ)
    cos_2α, sin_2α = cos(2*α), sin(2*α)
    sin_β_sq = sin(β)**2

    # Leading term (hVBaL)
    term_1 = 2*q*r*z*(5*p4 + (r2 + z2)**2 + 2*p2*(5*r2 + 3*z2) -
                      4*p*r*(3*p2 + r2 + z2)*cosφ)

    term_2 = -3*q*r*z*(-3*p4 + 2*p2*(r - z)*(r + z) + (r2 + z2)**2 +
                       4*p*r*(p2 - r2 - z2)*cosφ)

    term_3 = -r*z*(-11*p8 - 8*p2*(r2 - 4*z2)*(r2 + z2)**2 + (r2 + z2)**4 +
                   8*p6*(2*r2 + z2) + 2*p4*(r4 + 18*r2*z2 + 25*z4) +
                   4*p*r*(5*p6 + p2*(7*r2 - 13*z2)*(r2 + z2) - (r2 + z2)**3 -
                         p4*(11*r2 + 7*z2))*cosφ)

    term_4 = -8*p*r*z2*(4*p6 - p4*(r2 - 7*z2) - (r2 + z2)**3 +
                        2*p2*(-r4 + z4) +
                        p*r*(-7*p4 + 6*p2*(r - z)*(r + z) + (r2 + z2)**2)*cosφ)

    frac = sinφ/(4*q**2*(p2 + r2 + z2 - 2*p*r*cosφ)**(3/2))
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VBaL = frac * numer

    # Second term (hVBaF) - elliptic integral of first kind
    frac = z/(4*p*q*sqrt((p - r)**2 + z2))
    term_1 = -2*q
    term_2 = 3*q
    term_3 = (p4 - 2*p2*(r2 - 3*z2) + (r2 + z2)**2)
    term_4 = 2*p*z*(p2 - r2 - z2)
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VBaF = frac * numer * ellipfinc(varφ, m)

    # Third term (hVBaE) - elliptic integral of second kind
    frac = z*sqrt((p - r)**2 + z2)/(4*p*q**2)  # Changed from 8 to 4
    term_1 = 2*q*(3*p2 + r2 + z2)              # Changed from 4q to 2q
    term_2 = 3*q*(p2 - r2 - z2)                # Changed from 6q to 3q
    term_3 = (5*p6 + p2*(7*r2 - 13*z2)*(r2 + z2) - (r2 + z2)**3 -
              p4*(11*r2 + 7*z2))                # Changed from 2*(...) to (...)
    term_4 = 2*p*z*(-7*p4 + 6*p2*(r - z)*(r + z) + (r2 + z2)**2)  # Changed from 4p to 2p
    numer = term_1 + (term_2 + term_3*cos_2α + term_4*sin_2α)*sin_β_sq
    h_VBaE = frac * numer * ellipeinc(varφ, m)

    return h_VBaL + h_VBaF + h_VBaE

# fmt: on
