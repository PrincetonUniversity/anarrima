#!/usr/bin/env python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import grad, jit
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)

import math

from anarrima.reduced_integrals import g_HA, g_HB, g_HI, g_VA, g_VB, g_VI
from anarrima.visible_angle import _p0p1, visible_angle, wall_point

g_HI = jnp.vectorize(g_HI)
g_VI = jnp.vectorize(g_VI)

g_HA = jnp.vectorize(g_HA)
g_VA = jnp.vectorize(g_VA)

g_HB = jnp.vectorize(g_HB)
g_VB = jnp.vectorize(g_VB)


# Number of points
n_modes = 9
n_points = 48

# Initial guess: a circle in the r-z plane
initial_R0 = 0.20
initial_a = 0.1
initial_guess = np.concatenate(
    [np.array([initial_R0, initial_a]), np.zeros(n_modes - 1)]
)

plasma_point = (0.20, 0.0)


def generate_bounds(nwl, n_modes=n_modes):
    typ = 2.0 * 0.05 * math.sqrt(10 / nwl)
    return [(0.16, 0.30), (1e-8, typ)] + [
        (-typ / nc**2, typ / nc**2) for nc in range(1, n_modes)
    ]


def generate_guess(nwl, n_modes=n_modes):
    typ = 0.05 * math.sqrt(10 / nwl)
    initial_guess = np.concatenate([np.array([initial_R0, typ]), np.zeros(n_modes - 1)])
    return initial_guess


bounds = generate_bounds(2.5)


def plot_an_x(ax, x, n_points=n_points, **kwargs):
    r, z = unpack_variables(x, n_points)
    r_plot = np.append(r, r[0])
    z_plot = np.append(z, z[0])
    ax.plot(r_plot, z_plot, **kwargs)


def distribute_angles_for_radii(n_points):
    return np.linspace(0, 2 * np.pi, num=n_points, endpoint=False)


def get_radii(x, n_points=n_points):
    """Convert flat array to set of radii"""
    r0 = x[0]
    amplitude_cosine_modes = x[1:]
    angle = distribute_angles_for_radii(n_points)
    radii = jnp.sum(
        jnp.array(
            [amplitude_cosine_modes[m] * jnp.cos(m * angle) for m in range(n_modes)]
        ),
        axis=0,
    )
    return radii


def unpack_variables(x, n_points=n_points, n_modes=n_modes):
    """Convert flat array to r and z coordinates."""
    r0 = x[0]
    amplitude_cosine_modes = x[1:]
    angle = distribute_angles_for_radii(n_points)
    radii = jnp.sum(
        jnp.array(
            [amplitude_cosine_modes[m] * jnp.cos(m * angle) for m in range(n_modes)]
        ),
        axis=0,
    )

    r = r0 + radii * jnp.cos(angle)
    z = radii * jnp.sin(angle)
    return r, z


def two_next_differences(v):
    v0 = v
    v1 = jnp.roll(v, -1)
    v2 = jnp.roll(v, -2)
    w0 = v1 - v0
    w1 = v2 - v1
    return w0, w1


def convexity_constraint(x):
    """
    Ensure that each segment is "to the left" of the previous segment,
    which ensures convexity in a counterclockwise ordering.

    Returns an array of values that should all be >= 0 for the constraint to be satisfied.
    """
    r, z = unpack_variables(x)
    v1_r, v2_r = two_next_differences(r)
    v1_z, v2_z = two_next_differences(z)
    cross_z = v1_r * v2_z - v1_z * v2_r

    return cross_z


def positive_radius_constraint(x):
    r, _ = unpack_variables(x)
    return r


convexity_constraint_jac = jax.jacrev(convexity_constraint)
positive_radius_constraint_jac = jax.jacrev(positive_radius_constraint)


# Set up the constraints
constraints = [
    {
        "type": "ineq",
        "fun": convexity_constraint,
        "jac": convexity_constraint_jac,
    },  # Convexity constraints
    {
        "type": "ineq",
        "fun": positive_radius_constraint,
        "jac": positive_radius_constraint_jac,
    },  # positive radius
]


def wall_centers(r, z):
    wr, wz = jnp.array([wall_point(r, z, i, 0.5) for i in range(len(r))]).T
    return wr, wz


def wall_normals(r, z):
    r0, r1, z0, z1 = _p0p1(r, z)
    dr = r1 - r0
    dz = z1 - z0
    # watch out for zero-length segments!
    norm = jnp.sqrt(dr**2 + dz**2)
    dr_norm = dr / norm
    dz_norm = dz / norm
    return -dz_norm, dr_norm


@jit
def visible_angles(r, z, p):
    rp, zp = p
    ϕ = jnp.array([visible_angle(r, z, i, 0.5, rp, zp) for i in range(len(r))])
    return ϕ


def safe_arccos(cosϕ):
    ok_cosϕ = (-1.0 <= cosϕ) & (cosϕ <= 1.0)
    safe_cosϕ = -1.0
    fixed_cosϕ = jnp.where(ok_cosϕ, cosϕ, safe_cosϕ)
    ϕ = jnp.where(ok_cosϕ, jnp.arccos(fixed_cosϕ), jnp.pi)
    return ϕ


def construct_nwl_f(g_H, g_V):
    def f(r, z, plasma_point):
        p, zp_abs = plasma_point
        cosϕ = visible_angles(r, z, plasma_point)
        ϕ = safe_arccos(cosϕ)

        wr, wz = wall_centers(r, z)
        nr, nz = wall_normals(r, z)

        zp_rel = zp_abs - wz

        # g_HI: horizontal component, isotropic
        horiz = nr * g_H(p, zp_rel, wr, ϕ)
        vert = nz * g_V(p, zp_rel, wr, ϕ)
        return horiz + vert

    return f


nwl_iso = construct_nwl_f(g_HI, g_VI)
nwl_A = construct_nwl_f(g_HA, g_VA)
nwl_B = construct_nwl_f(g_HB, g_VB)


def f_of_nwl(x, desired_nwl):
    return jnp.sum((x - desired_nwl) ** 2)


def objective_iso(x, desired_nwl):
    r, z = unpack_variables(x)
    nwl = (1 / 3) * nwl_iso(r, z, plasma_point)
    return f_of_nwl(nwl, desired_nwl)


grad_iso = grad(objective_iso)

jit_obj_iso = jit(objective_iso)
obj_iso_jac = jit(grad(objective_iso))


def objective_A(x, desired_nwl):
    r, z = unpack_variables(x)
    nwl = (1 / 2) * nwl_A(r, z, plasma_point)
    return f_of_nwl(nwl, desired_nwl)


jit_obj_A = jit(objective_A)
obj_A_jac = jit(grad(objective_A))


def objective_B(x, desired_nwl):
    r, z = unpack_variables(x)
    nwl = (2 / 3) * nwl_B(r, z, plasma_point)
    return f_of_nwl(nwl, desired_nwl)


jit_obj_B = jit(objective_B)
obj_B_jac = jit(grad(objective_B))


def optimize_with_f(obj, jac, nwl_desired, initial_guess=None):
    # Run the optimization
    bounds = generate_bounds(nwl_desired)
    if initial_guess is None:
        initial_guess = generate_guess(nwl_desired)

    result = minimize(
        obj,
        initial_guess,
        args=(nwl_desired,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        jac=jac,
        options={"maxiter": 1000},
    )
    return result.x, result


# Generate the NWL target values. Use a geometric spacing.
# On the low end, any lower and the A-mode cannot be solved:
# the wall would have to retreat ever further inward,
# and there is a minimum NWL even when the center stack becomes like a spaghetti filament!

# On the high end, the shape becomes like a very-slightly offset circle.
loadings = np.round(np.geomspace(5.5, 15, num=10), 1)


xs_A = {}
previous = None
for wl in loadings[::-1]:
    xs_A[float(wl)], _ = optimize_with_f(
        jit_obj_A, obj_A_jac, wl, initial_guess=previous
    )
    previous = xs_A[float(wl)]
    print(f"A mode {float(wl)}: " + _.message)


xs_B = {}
previous = None
for wl in loadings[::-1]:
    xs_B[float(wl)], _ = optimize_with_f(
        jit_obj_B, obj_B_jac, wl, initial_guess=previous
    )
    previous = xs_B[float(wl)]
    print(f"Isotropic mode {float(wl)}: " + _.message)


xs_iso = {}
previous = None
for wl in loadings[::-1]:
    xs_iso[float(wl)], _ = optimize_with_f(
        jit_obj_iso, obj_iso_jac, wl, initial_guess=previous
    )
    previous = xs_iso[float(wl)]
    print(f"B/C modes {float(wl)}: " + _.message)


fig, axs = plt.subplots(1, 3, figsize=(8, 4), layout="constrained", sharey=True)

for wl, a in xs_A.items():
    plot_an_x(axs[0], a, n_points=120, lw=0.5, color="red")

for wl, a in xs_iso.items():
    plot_an_x(axs[1], a, n_points=120, lw=0.5, color="black")

for wl, a in xs_B.items():
    plot_an_x(axs[2], a, n_points=120, lw=0.5, color="blue")

axs[0].set_ylabel("Z")

rp, zp = plasma_point
for ax in axs:
    ax.scatter([rp], [zp], 70, marker="*", color="black", zorder=2000)

    ax.set_aspect("equal")
    # ax.legend(loc='upper right', title="NWL (arb.)")
    ax.set_xlim(0.0, 0.40)
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlabel("R")

    ax.grid(color="#EEEEEE")
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4])

axs[0].set_title("A mode")
axs[1].set_title("Isotropic mode")
axs[2].set_title("B/C modes")
axs[1].text(0.2, 0.125, r"5.5")
axs[1].text(0.2, 0.020, "15")

plt.savefig("nwl_ring_shells.pdf")
plt.show()


fig, axs = plt.subplots(1, 3, figsize=(10, 4), layout="constrained", sharey=True)

n_points_show = 120
θ = distribute_angles_for_radii(n_points_show)

for wl, x in xs_A.items():
    axs[0].plot(
        θ,
        (1 / 2) * nwl_A(*unpack_variables(x, n_points=n_points_show), plasma_point),
        lw=0.5,
        color="red",
    )

for wl, x in xs_iso.items():
    axs[1].plot(
        θ,
        (1 / 3) * nwl_iso(*unpack_variables(x, n_points=n_points_show), plasma_point),
        lw=0.5,
        color="black",
    )

for wl, x in xs_B.items():
    axs[2].plot(
        θ,
        (2 / 3) * nwl_B(*unpack_variables(x, n_points=n_points_show), plasma_point),
        lw=0.5,
        color="blue",
    )

axs[0].set_ylabel("Neutron Wall Load (unitless)")

for ax in axs:
    ax.set_xticks(
        [0, jnp.pi / 4, jnp.pi / 2, np.pi, 3 * np.pi / 2, 2 * jnp.pi],
        [0, 1 / 4, 0.5, 1, 1.5, 2],
    )
    ax.set_xlabel("θ around wall")
    ax.set_ylim(5, 16)
    ax.set_xlim(0, 2 * jnp.pi)
    ax.set_yscale("log")
    ax.set_yticks(
        [5.5, 6.1, 6.9, 7.7, 8.6, 9.6, 10.7, 12.0, 13.4, 15.0],
        labels=[5.5, 6.1, 6.9, 7.7, 8.6, 9.6, 10.7, 12.0, 13.4, 15.0],
    )
    ax.set_xticks([0.0, jnp.pi / 2, jnp.pi, 3 * jnp.pi / 2, 2 * jnp.pi])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax.set_yticks([], minor=True)

axs[0].set_title("A mode")
axs[1].set_title("Isotropic mode")
axs[2].set_title("B/C modes")
plt.savefig("nwl_ring_shells_verification.pdf")

plt.show()
