import numpy as np
from scipy.special import ellipeinc, ellipkinc

def amplitude_and_parameter(p, z, r, φ):
    varφ = φ/2
    m = - 4 * p * r / ((p - r)**2 + z**2)
    return varφ, m

def q_term(p, z, r):
    return p**4 - 2 * p**2 * (r**2 - z**2) + (r**2 + z**2)**2

#########################
# Isotropic horizontal
#########################

def h_HIL_term(p, z, r, φ):
    q = q_term(p, z, r)
    numer = 4 * p**2 * (p**2 - r**2 + z**2) * np.sin(φ)
    denom  = q * np.sqrt(p**2 + r**2 + z**2 - 2 * p * r * np.cos(φ))
    return numer/denom

def h_HIE_term(p, z, r, φ):
    q = q_term(p, z, r)
    numer = 2 * p * np.sqrt((p-r)**2 + z**2) * (p**2 - r**2 + z**2)
    denom = q * r
    return numer/denom

def h_HIK_term(p, z, r, φ):
    numer = -2 * p
    denom = r * np.sqrt((p - r)**2 + z**2)
    return numer/denom

def g_HI(p, z, r, φ):
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    h_HIL = h_HIL_term(p, z, r, φ)
    h_HIE = h_HIE_term(p, z, r, φ) * ellipeinc(varφ, m)
    h_HIK = h_HIK_term(p, z, r, φ) * ellipkinc(varφ, m)
    return h_HIL + h_HIE + h_HIK

#########################
# Isotropic vertical
#########################

def g_VI(p, z, r, φ):
    # L term
    numer = 8 * p**2 * r * z * np.sin(φ)
    denom = (((p**2 + r**2 + z**2)**2 - 4*p**2*r**2) *
                   np.sqrt(p**2 + r**2 + z**2 - 2*p*r*np.cos(φ)))
    h_VIL = numer / denom

    # E term
    varφ, m = amplitude_and_parameter(p, z, r, φ)
    numer = 4 * p * z * ellipeinc(varφ, m)
    denom = np.sqrt((p - r)**2 + z**2) * ((p + r)**2 + z**2)
    h_VIE = numer / denom

    return h_VIL + h_VIE
