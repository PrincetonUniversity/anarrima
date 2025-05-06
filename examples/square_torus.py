from anarrima import reduced_integrals
import numpy as np
import matplotlib.pyplot as plt

#### Locations of walls

u_inboard = 0.4
w_outboard = 1.6
z_topbot = 0.6

### Plasma source grid


def parabolic_profile(r):
    """Generic 'parabolic' profile shape"""
    return 1 - r**2


def fusion_rate(r):
    return parabolic_profile(r)


minor_radius = 0.5
major_radius = 1

source_grid_nr = 30
source_grid_nz = 30
source_grid_n = source_grid_nr * source_grid_nz

source_axis_r = np.linspace(0.5, 1.5, source_grid_nr)
source_axis_z = np.linspace(-0.5, 0.5, source_grid_nz)
source_Δr = (max(source_axis_r) - min(source_axis_r)) / source_grid_nr
source_Δz = (max(source_axis_z) - min(source_axis_z)) / source_grid_nz

source_p, source_zabs = np.meshgrid(source_axis_r, source_axis_z)


def grid_to_column(g):
    return g.reshape(source_grid_n)


minor_radius_sq_at_point = (source_p - major_radius) ** 2 + source_zabs**2
minor_radii_sq = grid_to_column(minor_radius_sq_at_point)

# test which points are within the LCFS
in_lcfs = minor_radii_sq < minor_radius**2

source_p_array = grid_to_column(source_p)
source_zabs_array = grid_to_column(source_zabs)

valid_source_points = np.array([source_p_array, source_zabs_array]).T[in_lcfs]

fusion_grid = (
    np.where(
        minor_radius_sq_at_point < minor_radius**2,
        fusion_rate(np.sqrt(minor_radius_sq_at_point) / minor_radius),
        0.0,
    )
    * source_Δr
    * source_Δz
)


z_wall = np.linspace(-z_topbot, z_topbot, 200)
v_wall = np.linspace(u_inboard, w_outboard, 200)
z_wall_Δz = (max(z_wall) - min(z_wall)) / len(z_wall)
v_wall_Δv = (max(v_wall) - min(v_wall)) / len(v_wall)

### Only perform computations for plasma rings within the LCFS

p = source_p_array[in_lcfs]
zz = source_zabs_array[in_lcfs]
strength = grid_to_column(fusion_grid)[in_lcfs]


## NWL on inboard wall

ϕ_i = np.arccos(u_inboard / p)


def irrad_inboard(z_wall):
    z_rel = z_wall - zz
    g = reduced_integrals.g_HI(p, z_rel, u_inboard, ϕ_i)
    return np.dot(g, strength)


def irrad_inboard_A(z_wall):
    z_rel = z_wall - zz
    g = reduced_integrals.g_HA(p, z_rel, u_inboard, ϕ_i)
    return np.dot(g, strength)


def irrad_inboard_B(z_wall):
    z_rel = z_wall - zz
    g = reduced_integrals.g_HBa(p, z_rel, u_inboard, 0, 0, ϕ_i)
    return np.dot(g, strength)


## NWL on floor/ceiling

z_rel_floor = zz - (-z_topbot)


def ϕ_floor(v):
    return np.arccos(u_inboard / p) + np.arccos(u_inboard / v)


def irrad_floor(v_wall):
    ϕ_max = ϕ_floor(v_wall)
    g = reduced_integrals.g_VI(p, z_rel_floor, v_wall, ϕ_max)
    return np.dot(g, strength)


def irrad_floor_A(v_wall):
    ϕ_max = ϕ_floor(v_wall)
    g = reduced_integrals.g_VAa(p, z_rel_floor, v_wall, 0, 0, ϕ_max)
    return np.dot(g, strength)


def irrad_floor_B(v_wall):
    ϕ_max = ϕ_floor(v_wall)
    g = reduced_integrals.g_VBa(p, z_rel_floor, v_wall, 0, 0, ϕ_max)
    return np.dot(g, strength)


## NWL on outboard wall

ϕ_o = np.arccos(u_inboard / p) + np.arccos(u_inboard / w_outboard)


def irrad_outboard(z_wall):
    z_rel = zz - z_wall
    g = -reduced_integrals.g_HI(p, z_rel, w_outboard, ϕ_o)
    return np.dot(g, strength)


def irrad_outboard_A(z_wall):
    z_rel = zz - z_wall
    g = -reduced_integrals.g_HAa(p, z_rel, w_outboard, 0, 0, ϕ_o)
    return np.dot(g, strength)


def irrad_outboard_B(z_wall):
    z_rel = zz - z_wall
    g = -reduced_integrals.g_HBa(p, z_rel, w_outboard, 0, 0, ϕ_o)
    return np.dot(g, strength)


### Compute NWL from different modes
#
# σ = σ₀/2π (3/4 a sin²θ + (2/3 b + 1/3 c)(1 + 3 cos²θ)/4)
# here 'I' means isotropic, for the unpolarized mixture: a=b=c=1/3

fI_inboard = np.array([(1 / 3) * irrad_inboard(zw) for zw in z_wall])
fa_inboard = np.array([(3 / 4) * irrad_inboard_A(zw) for zw in z_wall])
fb_inboard = np.array([(2 / 3) * irrad_inboard_B(zw) for zw in z_wall])

fI_floor = np.array([(1 / 3) * irrad_floor(v) for v in v_wall])
fa_floor = np.array([(3 / 4) * irrad_floor_A(v) for v in v_wall])
fb_floor = np.array([(2 / 3) * irrad_floor_B(v) for v in v_wall])

fI_outboard = np.array([(1 / 3) * irrad_outboard(zw) for zw in z_wall])
fa_outboard = np.array([(3 / 4) * irrad_outboard_A(zw) for zw in z_wall])
fb_outboard = np.array([(2 / 3) * irrad_outboard_B(zw) for zw in z_wall])

### Functions for plotting with the wall serving as the baseline of a plot
norm = 1


def plot_inboard(ax, z, val, **kwargs):
    ax.plot(u_inboard - val / norm, z, **kwargs)


def plot_outboard(ax, z, val, **kwargs):
    ax.plot(w_outboard + val / norm, z, **kwargs)


def plot_floor_ceil(ax, v, val, **kwargs):
    ax.plot(v, -z_topbot - val / norm, **kwargs)
    ax.plot(v, z_topbot + val / norm, **kwargs)


def plot_walls(ax, **kwargs):
    """Four sides of the rectangular torus cross section"""
    r = [u_inboard, w_outboard, w_outboard, u_inboard, u_inboard]
    z = [-z_topbot, -z_topbot, z_topbot, z_topbot, -z_topbot]
    ax.plot(r, z, **kwargs)


### Make a plot
fig, ax = plt.subplots()

# Plot the plasma. Hide points outside the LCFS
outside_lcfs = ~in_lcfs.reshape(source_grid_nr, source_grid_nz)
fusion_grid[outside_lcfs] = np.nan
ax.contourf(source_p, source_zabs, fusion_grid, cmap="plasma")

plot_walls(ax, lw=1.5, color="gray")

### Plot NWL from isotropic, A-mode, and B/C-mode mixes
# normalize the A-mode by 2/3 to better show the distribution of neutron fluxes
# while keeping the total fusion power fixed.
style_I = {"color": "#333333", "ls": "dashed"}
style_A = {"color": "red"}
style_B = {"color": "blue"}

plot_inboard(ax, z_wall, fI_inboard, **style_I, label="Isotropic")
plot_inboard(ax, z_wall, (2 / 3) * fa_inboard, **style_A, label="A-mode * (2/3)")
plot_inboard(ax, z_wall, fb_inboard, **style_B, label="B/C-mode")

plot_floor_ceil(ax, v_wall, fI_floor, **style_I)
plot_floor_ceil(ax, v_wall, (2 / 3) * fa_floor, **style_A)
plot_floor_ceil(ax, v_wall, fb_floor, **style_B)

plot_outboard(ax, z_wall, fI_outboard, **style_I)
plot_outboard(ax, z_wall, (2 / 3) * fa_outboard, **style_A)
plot_outboard(ax, z_wall, fb_outboard, **style_B)

extra_plotrange = 0.6
ax.set_ylim(-z_topbot - extra_plotrange, +z_topbot + 1.5 * extra_plotrange)
ax.set_xlim(u_inboard - extra_plotrange, w_outboard + 1.5 * extra_plotrange)

[sp.set_visible(False) for _, sp in ax.spines.items()]
ax.set_xticks([])
ax.set_yticks([])

ax.set_aspect(1)
ax.legend()
ax.annotate("", xytext=(0, -1), xy=(0, 1), arrowprops=dict(arrowstyle="->", lw=0.5))

plt.show()
