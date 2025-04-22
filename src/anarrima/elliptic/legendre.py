from anarrima.elliptic.carlson import rf, rd
import jax
import jax.numpy as jnp

sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos

def ellipk(m):
    return rf(0., 1-m, 1.)

@jax.custom_jvp
def ellipfinc(φ, m):
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * rf(x, y, z)

@jax.custom_jvp
def ellipeinc(φ, m):
    sinφ = sin(φ)
    sin_sq_φ = jnp.square(sinφ)
    sin_cu_φ = sin_sq_φ * sinφ
    x = 1. - sin_sq_φ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * rf(x, y, z) - m * sin_cu_φ * rd(x, y, z) / 3

@jax.jit
def ellip_f_e_fused(φ, m):
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
@ellipfinc.defjvp
def ellipfinc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents

    finc, einc = ellip_f_e_fused(φ, m)

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
    finc, einc = ellip_f_e_fused(φ, m)
    primal_out = einc

    d_ellipe_dφ = sqrt(1 - m*sin(φ)**2)
    d_ellipe_dm = (einc - finc)/(2*m)

    tangent_out = d_ellipe_dφ * φ_dot + d_ellipe_dm * m_dot
    return primal_out, tangent_out
