import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from pytest import approx

INF = jnp.inf
PI2 = jnp.pi/2
PI4 = jnp.pi/4
NAN = jnp.nan

isnan = jnp.isnan
isinf = jnp.isinf

# Helper function to check if values match, handling special cases
def values_match(a, b):
    if isnan(a) and isnan(b):
        return True
    elif isinf(a) and isinf(b):
        return jnp.sign(a) == jnp.sign(b)
    else:
        # For finite values, use approximate comparison
        return a == approx(b, rel=1e-15)
