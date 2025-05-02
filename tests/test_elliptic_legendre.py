import anarrima.elliptic.legendre as legendre
import pytest
from pytest import approx
import jax.numpy as jnp
from jax import grad

einc = legendre.ellipeinc
finc = legendre.ellipfinc

### Ellipfinc tests
φ0 = jnp.pi/4
m0 = -1.
  
## Gradients of ellipf

# def test_fused_form_equality():
#     f1 = finc(φ0, m0)
#     e1 = einc(φ0, m0)
#     f2, e2 = legendre.ellip_finc_einc_fused(φ0, m0)
#     assert f1 == f2
#     assert e1 == e2

