from scipy.special import ellipeinc as scipy_ellipeinc
from scipy.special import ellipkinc as scipy_ellipfinc # for consistency with paper
import jax.numpy as jnp
import jax

sqrt = jnp.sqrt
sin = jnp.sin
cos = jnp.cos

@jax.custom_jvp
def ellipfinc(varφ, m):
    """Incomplete elliptic integral of the first kind
    """
    return scipy_ellipfinc(varφ, m)

@jax.custom_jvp
def ellipeinc(varφ, m):
    """Incomplete elliptic integral of the second kind
    """
    return scipy_ellipeinc(varφ, m)

# This will fail to evaluate if m==0 or m==1.
# However, since m = -4 p r / ((p-r)² + z²) it is strictly negative
# for physically meaningful values of m
@ellipfinc.defjvp
def ellipfinc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents
    primal_out = ellipfinc(φ, m)

    d_ellipf_dφ = 1/sqrt(1 - m*sin(φ)**2)

    # compute dK/dm
    d_L = sin(2 * φ) / (4 * (m-1) * sqrt(1-m*sin(φ)**2))
    d_E = -ellipeinc(φ,m)/(2 * (m-1)*m)
    d_F = -ellipfinc(φ,m)/(2 * m)
    d_ellipf_dm = d_L + d_E + d_F

    tangent_out = d_ellipf_dφ * φ_dot + d_ellipf_dm * m_dot
    return primal_out, tangent_out

@ellipeinc.defjvp
def ellipeinc_jvp(primals, tangents):
    φ, m = primals
    φ_dot, m_dot = tangents
    primal_out = ellipeinc(φ, m)

    d_ellipe_dφ = sqrt(1 - m*sin(φ)**2)
    d_ellipe_dm = (ellipeinc(φ, m) - ellipfinc(φ,m))/(2*m)

    tangent_out = d_ellipe_dφ * φ_dot + d_ellipe_dm * m_dot
    return primal_out, tangent_out

