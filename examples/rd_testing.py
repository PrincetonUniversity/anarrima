import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit
from jax.lax import scan
from scipy.special import elliprd as scipy_rd
from scipy.special import elliprf as scipy_rf

from anarrima.elliptic.carlson import _rd as anarrima_rdn
from anarrima.elliptic.carlson import rd as anarrima_rd

sqrt = jnp.sqrt


def plot_logrel_error(ax, logrel):
    pos = ax.imshow(logrel, cmap="Blues", interpolation="none")
    fig.colorbar(pos, ax=ax)


_RD_NUM_LOOPS = 9


def rd_plain(x0, y0, z0):
    """ """

    NUM_LOOPS = _RD_NUM_LOOPS

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

    numerators = jnp.power(4.0, -jnp.arange(NUM_LOOPS))

    result, sum_elements = scan(body, init, xs=numerators, length=NUM_LOOPS)
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


# # Accuracy verification


decades = 18
n_square = 50
x_list = jnp.logspace(-decades, decades, n_square)
y_list = jnp.logspace(-decades, decades, n_square)
xx, yy = jnp.meshgrid(x_list, y_list)
zz = jnp.ones_like(xx)


vec_rd_plain = jit(jnp.vectorize(rd_plain))
vec_rd_anarr = jit(jnp.vectorize(anarrima_rd))
vec_rd_anarrn = jit(jnp.vectorize(anarrima_rdn, excluded=(3,)), static_argnums=3)
vec_g_rd_plain = jit(jnp.vectorize(grad(rd_plain)))
vec_g_rd_anarr = jit(jnp.vectorize(grad(anarrima_rd)))

test_plain = vec_g_rd_plain(xx, yy, zz)
test_anarr = vec_g_rd_anarr(xx, yy, zz)

ref = scipy_rd(xx, yy, zz)
test = vec_rd_anarr(xx, yy, zz)

rel = jnp.abs((test - ref) / ref)
logrel = jnp.log10(rel)


fig, ax = plt.subplots()
plot_logrel_error(ax, logrel)
ax.set_title("Log10 of rel. accuracy of Rd")
plt.show()

# Wide-range testing

decades = 150
n_square = 1000
x_list = jnp.zeros(n_square)
y_list = jnp.logspace(-decades, decades, n_square)
z_list = jnp.ones_like(x_list)

ref = scipy_rd(x_list, y_list, z_list)
test = vec_rd_anarr(x_list, y_list, z_list)


def logrel_of_iter(n):
    test_n = vec_rd_anarrn(x_list, y_list, z_list, n)
    rel = jnp.abs((test_n - ref) / ref)
    logrel = jnp.log10(rel)
    return logrel


fig, ax = plt.subplots()
# ax.plot(y_list, logrel_of_iter(5))
# ax.plot(y_list, logrel_of_iter(6))
# ax.plot(y_list, logrel_of_iter(7))
ax.plot(y_list, logrel_of_iter(8))
ax.plot(y_list, logrel_of_iter(9))
ax.plot(y_list, logrel_of_iter(10))

# ax.set_yscale('log')
ax.set_xscale("log")
ax.set_title("Log10 of accuracy of Rd over wide range")
plt.show()


# # Time testing

vec_rd_plain = jit(jnp.vectorize(rd_plain))


# %%timeit
# ref = scipy_rf(xx, yy, zz)


# %%timeit
# test = vec_rf_plain(xx, yy, zz).block_until_ready()


# # Grad time testing


vec_g_rd_plain = jit(jnp.vectorize(grad(rd_plain)))
vec_g_rd_anarr = jit(jnp.vectorize(grad(anarrima_rd)))
test = vec_g_rd_plain(xx, yy, zz).block_until_ready()


# %%timeit
# test = vec_g_rd_plain(xx, yy, zz).block_until_ready()


# %%timeit
# test = vec_g_rd_anarr(xx, yy, zz).block_until_ready()


# ### Grad accuracy testing


test_plain = vec_g_rd_plain(xx, yy, zz).block_until_ready()
test_anarr = vec_g_rd_anarr(xx, yy, zz).block_until_ready()

rel = jnp.abs((test_plain - test_anarr) / test_plain)
logrel = jnp.log10(rel)

fig, ax = plt.subplots()
plot_logrel_error(ax, logrel)
ax.set_title("Log10 of accuracy of d/dx (Rd)")
plt.show()
