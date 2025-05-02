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
  
