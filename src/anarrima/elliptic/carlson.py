import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.lax import scan
from jax import custom_jvp
from functools import partial

sqrt = jnp.sqrt

def rf(x, y, z):
    return _rf(x, y, z, 8)

def rf_quick(x, y, z):
    return _rf(x, y, z, 3)

@partial(custom_jvp, nondiff_argnums=(3,))
def _rf(x0, y0, z0, NUM_LOOPS):

    v0 = jnp.array([x0, y0, z0])
    A0 = jnp.sum(v0) / 3
    init = A0, v0

    def body(carry, _):
        Am, vm = carry
        λ = sqrt(vm[0]*vm[1]) + sqrt(vm[0]*vm[2]) + sqrt(vm[1]*vm[2])

        Am_new = (Am + λ) / 4
        vm_new = (vm + λ) / 4

        return (Am_new, vm_new), None

    result, _ = scan(body, init, length=NUM_LOOPS)
    an, _ = result

    f = 4**(-NUM_LOOPS)

    x = (A0 - x0) / an * f
    y = (A0 - y0) / an * f
    z = -(x + y)
    e2 = x * y - z * z
    e3 = x * y * z

    return ( 1 - e2 / 10
               + e3 / 14
               + e2 * e2 / 24
               - 3 * e2 * e3 / 44
               - 5 * e2**3 / 208 
               + 3 * e3**2 / 104) / sqrt(an)

def rd(x0, y0, z0):
    return _rd(x0, y0, z0, 8)

def _rd(x0, y0, z0, NUM_LOOPS):
    """
    """
    v0 = jnp.array([x0, y0, z0])
    A0 = (x0 + y0 + 3*z0) / 5
    init = A0, v0

    def body(carry, numer):
        Am, vm = carry
        xm, ym, zm = vm

        r_xm, r_ym, r_zm = sqrt(xm), sqrt(ym), sqrt(zm)
        λ = r_xm * r_ym + r_ym * r_zm + r_zm * r_xm

        Am_new = (Am + λ) / 4
        vm_new = (vm + λ) / 4

        denom = r_zm * (zm + λ)

        sum_element = numer / denom

        return (Am_new, vm_new), sum_element

    numerators = jnp.power(4., -jnp.arange(NUM_LOOPS))

    result, sum_elements = scan(body, init, xs=numerators, length=NUM_LOOPS)
    an, _ = result

    scale = numerators[-1]/4

    x = (A0 - x0) / an * scale
    y = (A0 - y0) / an * scale
    z = -(x + y) / 3

    e2 = x * y - 6 * z**2
    e3 = (3 * x * y - 8 * z**2) * z
    e4 = 3 * (x * y - z**2) * z**2
    e5 = x * y * z**3

    c1, c2, c3 = -3/14, 1/6, 9/88
    c4, c5, c6 = -3/22, -9/52, 3/26

    series = (1 + c1 * e2 + c2 * e3
                + c3 * e2**2 + c4 * e4
                + c5 * e2 * e3 + c6 * e5)
    series_term = scale * series / (an * sqrt(an))

    sum_term = 3 * jnp.sum(sum_elements)

    return sum_term + series_term

@_rf.defjvp
def _rf_jvp(NUM_LOOPS, primals, tangents):
    x, y, z = primals
    x_dot, y_dot, z_dot = tangents
    primals_out = _rf(x, y, z, NUM_LOOPS)
    tangents_out = (-_rd(y, z, x, NUM_LOOPS)/6 * x_dot
                    -_rd(z, x, y, NUM_LOOPS)/6 * y_dot
                    -_rd(x, y, z, NUM_LOOPS)/6 * z_dot)
    return primals_out, tangents_out


# drfdx = lambda x_dot, primal_out, x, y, z: -rd(y, z, x)/6 * x_dot
# drfdy = lambda y_dot, primal_out, x, y, z: -rd(z, x, y)/6 * y_dot
# drfdz = lambda z_dot, primal_out, x, y, z: -rd(x, y, z)/6 * z_dot
# rf.defjvps(drfdx, drfdy, drfdz)

# rf_quick.defjvps(lambda x_dot, primal_out, x, y, z: -rd(y, z, x)/6 * x_dot,
#                  lambda y_dot, primal_out, x, y, z: -rd(z, x, y)/6 * y_dot,
#                  lambda z_dot, primal_out, x, y, z: -rd(x, y, z)/6 * z_dot)
