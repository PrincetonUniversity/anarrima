import jax

jax.config.update("jax_enable_x64", True)
from functools import partial

import jax.numpy as jnp
from jax import custom_jvp
from jax.lax import scan

sqrt = jnp.sqrt


# There are analytic forms for
# x, x, z or x, y, y but those cannot happen
# in the anarrima geometry; thus we only
# require the general case
def rf(x, y, z):
    return _rf(x, y, z, 8)


@partial(custom_jvp, nondiff_argnums=(3,))
def _rf(x0, y0, z0, n_loops):

    v0 = jnp.array([x0, y0, z0])
    A0 = jnp.sum(v0) / 3
    init = A0, v0

    def body(carry, _):
        Am, vm = carry
        λ = sqrt(vm[0] * vm[1]) + sqrt(vm[0] * vm[2]) + sqrt(vm[1] * vm[2])

        Am_new = (Am + λ) / 4
        vm_new = (vm + λ) / 4

        return (Am_new, vm_new), None

    result, _ = scan(body, init, length=n_loops)
    an, _ = result

    f = 4 ** (-n_loops)

    x = (A0 - x0) / an * f
    y = (A0 - y0) / an * f
    z = -(x + y)
    e2 = x * y - z * z
    e3 = x * y * z

    return (
        1
        - e2 / 10
        + e3 / 14
        + e2 * e2 / 24
        - 3 * e2 * e3 / 44
        - 5 * e2**3 / 208
        + 3 * e3**2 / 104
    ) / sqrt(an)


@_rf.defjvp
def _rf_jvp(n_loops, primals, tangents):
    x, y, z = primals
    x_dot, y_dot, z_dot = tangents
    primals_out = _rf(x, y, z, n_loops)
    tangents_out = (
        -_rd(y, z, x, n_loops) / 6 * x_dot
        - _rd(z, x, y, n_loops) / 6 * y_dot
        - _rd(x, y, z, n_loops) / 6 * z_dot
    )
    return primals_out, tangents_out


def rd(x0, y0, z0):
    return _rd(x0, y0, z0, 8)


@partial(custom_jvp, nondiff_argnums=(3,))
def _rd(x0, y0, z0, n_loops):
    """ """
    v0 = jnp.array([x0, y0, z0])
    A0 = (x0 + y0 + 3 * z0) / 5
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

    numerators = jnp.power(4.0, -jnp.arange(n_loops))

    result, sum_elements = scan(body, init, xs=numerators, length=n_loops)
    an, _ = result

    scale = numerators[-1] / 4

    x = (A0 - x0) / an * scale
    y = (A0 - y0) / an * scale
    z = -(x + y) / 3

    e2 = x * y - 6 * z**2
    e3 = (3 * x * y - 8 * z**2) * z
    e4 = 3 * (x * y - z**2) * z**2
    e5 = x * y * z**3

    c1, c2, c3 = -3 / 14, 1 / 6, 9 / 88
    c4, c5, c6 = -3 / 22, -9 / 52, 3 / 26

    series = 1 + c1 * e2 + c2 * e3 + c3 * e2**2 + c4 * e4 + c5 * e2 * e3 + c6 * e5
    series_term = scale * series / (an * sqrt(an))

    sum_term = 3 * jnp.sum(sum_elements)

    return sum_term + series_term


def _drd_dxyz_all_equal(x):
    # drd_dx and drd_dy are the same
    v = -3.0 / (10 * x ** (5.0 / 2))
    # drd_dz has -9/10 instead of -3/10
    return v, v, 3 * v


# not used at the moment
def _drd_dx_xz_equal(x, y):
    drd_dx_xz_numer = 3 * ((5 * x - 2 * y) * jnp.sqrt(y) / jnp.square(x) - 3 * rf0)
    drd_dx_xz_denom = 16 * (x - y) ** 2
    drd_dx_xz = drd_dx_xz_numer / drd_dx_xz_denom
    return drd_dx_xz


# Does this jvp ever add accuracy or speed up evaluation?
@_rd.defjvp
def _rd_jvp(n_loops, primals, tangents):
    x, y, z = primals
    x_dot, y_dot, z_dot = tangents

    rd0 = _rd(x, y, z, n_loops)
    rf0 = _rf(x, y, z, n_loops)
    rd_x_last = _rd(y, z, x, n_loops)
    rd_y_last = _rd(x, z, y, n_loops)

    primals_out = rd0

    # special case to avoid divide-by-zero in the standard case
    three_equal = (x == y) & (y == z)
    drd_dx_all_equal, drd_dy_all_equal, drd_dz_all_equal = _drd_dxyz_all_equal(x)
    # We might also define special cases for x == z ≠ y and y == z != x,
    # or for extreme values like infinity,
    # but those are not needed for now.
    is_standard = jnp.logical_not(three_equal)

    x_minus_z = jnp.where(is_standard, x - z, 1.0)
    y_minus_z = jnp.where(is_standard, y - z, 1.0)

    drd_dx_standard = (-rd0 + rd_x_last) / (2 * x_minus_z)
    drd_dy_standard = (-rd0 + rd_y_last) / (2 * y_minus_z)

    numer = -3 * sqrt(x * y) + z ** (3 / 2) * (2 * (x + y - 2 * z) * rd0 + 3 * rf0)
    denom = 2 * z ** (3 / 2) * (x_minus_z) * (y_minus_z)
    drd_dz_standard = numer / denom

    drd_dx = jnp.where(is_standard, drd_dx_standard, drd_dx_all_equal)
    drd_dy = jnp.where(is_standard, drd_dy_standard, drd_dy_all_equal)
    drd_dz = jnp.where(is_standard, drd_dz_standard, drd_dz_all_equal)
    tangents_out = drd_dx * x_dot + drd_dy * y_dot + drd_dz * z_dot
    return primals_out, tangents_out
