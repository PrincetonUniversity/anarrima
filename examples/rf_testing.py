#!/usr/bin/env python
# coding: utf-8

import jax

jax.config.update("jax_enable_x64", True)
import statistics
import timeit

import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import grad, jit
from jax.lax import scan
from scipy.special import elliprd as scipy_rd
from scipy.special import elliprf as scipy_rf

from anarrima.elliptic.carlson import _rf as anarrima_rfn
from anarrima.elliptic.carlson import rf as anarrima_rf

sqrt = jnp.sqrt


def plot_logrel_error(ax, logrel):
    pos = ax.imshow(logrel, cmap="Blues", interpolation="none")
    fig.colorbar(pos, ax=ax)


decades = 18
n_square = 50
x_list = jnp.logspace(-decades, decades, n_square)
y_list = jnp.logspace(-decades, decades, n_square)
xx, yy = jnp.meshgrid(x_list, y_list)
zz = jnp.ones_like(xx)

vec_rf_anarr = jit(jnp.vectorize(anarrima_rf))
vec_rf_anarrn = jit(jnp.vectorize(anarrima_rfn, excluded=(3,)), static_argnums=3)
vec_g_rf_anarr = jit(jnp.vectorize(grad(anarrima_rf)))

test_anarr = vec_g_rf_anarr(xx, yy, zz)

ref = scipy_rf(xx, yy, zz)

# change the number of iterations here
test = vec_rf_anarrn(xx, yy, zz, 9)

rel = jnp.abs((test - ref) / ref)
logrel = jnp.log10(rel)

fig, ax = plt.subplots()
plot_logrel_error(ax, logrel)
ax.set_title("Rf accuracy: log10 of relative error vs scipy")
plt.show()

decades = 150
n_square = 1000
x_list = jnp.zeros(n_square)
y_list = jnp.logspace(-decades, decades, n_square)
z_list = jnp.ones_like(x_list)

ref = scipy_rf(x_list, y_list, z_list)
test = vec_rf_anarr(x_list, y_list, z_list)


def logrel_of_iter(n):
    test_n = vec_rf_anarrn(x_list, y_list, z_list, n)
    rel = jnp.abs((test_n - ref) / ref)
    logrel = jnp.log10(rel)
    return logrel


fig, ax = plt.subplots()
# ax.plot(y_list, logrel_of_iter(5))
# ax.plot(y_list, logrel_of_iter(6))
# ax.plot(y_list, logrel_of_iter(7))
ax.plot(y_list, logrel_of_iter(8), label="8: standard for anarrima")
ax.plot(y_list, logrel_of_iter(9), label="9")
ax.plot(y_list, logrel_of_iter(10), label="10")
ax.legend(title="Number of iterations\nfor Rf calculation")

# ax.set_yscale('log')
ax.set_xscale("log")
ax.set_title(
    r"Evaluating $R_F(0,\,y,\,1)$ over a wide range:"
    + "\nlog10 of relative error vs scipy"
)


# # Time testing


def time_a_function(f):
    # Run timing (similar to timeit's default: 7 runs, auto-determined loops)
    number_of_loops = 1000  # You can adjust this or let timeit auto-determine
    runs = 7

    times = []
    for _ in range(runs):
        time_taken = timeit.timeit(f, number=number_of_loops)
        times.append(time_taken / number_of_loops)  # Time per single execution

    # Calculate statistics
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    print(f"Mean: {1000 * mean_time:.6f} ms, std dev {1000 * std_time:3.3f} ms")


def scipy_rf_timing():
    scipy_rf(xx, yy, zz)


def vec_rf_anarr_timing():
    vec_rf_anarr(xx, yy, zz).block_until_ready()


print("Rf by scipy:")
time_a_function(scipy_rf_timing)

print("Rf by anarrima:")
time_a_function(vec_rf_anarr_timing)


# Grad time testing


vec_g_rf_anarr = jit(jnp.vectorize(grad(anarrima_rf)))


def vec_g_rf_anarr_timing():
    vec_g_rf_anarr(xx, yy, zz).block_until_ready()


print("Gradient of Rf, by anarrima:")
time_a_function(vec_rf_anarr_timing)

# Grad accuracy testing


def gradient_of_rf(x, y, z):
    return -(1 / 6) * scipy_rd(y, z, x)


grad_reference = gradient_of_rf(xx, yy, zz)
test_anarr = vec_g_rf_anarr(xx, yy, zz).block_until_ready()

rel = jnp.abs((grad_reference - test_anarr) / grad_reference)
logrel = jnp.log10(rel)

fig, ax = plt.subplots()
plot_logrel_error(ax, logrel)
ax.set_title("Log10 of error in gradient of (Rf) vs scipy")
plt.show()
