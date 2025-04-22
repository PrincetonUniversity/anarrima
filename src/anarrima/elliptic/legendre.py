from anarrima.elliptic.carlson import rf, rd
import jax.numpy as jnp

def ellipeinc(φ, m):
    sinφ = jnp.sin(φ)
    cosφ = jnp.cos(φ)
    sin_sq_φ = sinφ * sinφ
    sin_cu_φ = sin_sq_φ * sinφ
    x = cosφ * cosφ
    y = 1. - m * sin_sq_φ
    z = 1.
    return sinφ * rf(x, y, z) - m * sin_cu_φ * rd(x, y, z) / 3

def ellipfinc(φ, m):
    sinφ = jnp.sin(φ)
    cosφ = jnp.cos(φ)
    x = cosφ * cosφ
    y = 1. - m * sinφ * sinφ
    z = 1.
    return sinφ * rf(x, y, z)
