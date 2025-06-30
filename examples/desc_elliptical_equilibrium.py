#!/usr/bin/env python
# coding: utf-8

import desc.io
import numpy as np
from desc.equilibrium import Equilibrium
from desc.geometry import FourierRZToroidalSurface
from desc.plotting import plot_comparison, plot_grid
from desc.profiles import PowerSeriesProfile

surf2 = FourierRZToroidalSurface(
    R_lmn=[0.2, -0.05],
    modes_R=[(0, 0), (1, 0)],
    Z_lmn=[0.075],
    modes_Z=[(-1, 0)],
    NFP=1,
)

# Not sure exactly what this pressure is
pressure = PowerSeriesProfile([1.8e4, 0, -3.6e4, 0, 1.8e4])

# vaguely similar to a q = 1 to q=8 profile
iota = PowerSeriesProfile([6, 0, -9, 0, 4])  # 6 - 9 \[Rho]^2 + 4 \[Rho]^4
eq = Equilibrium(L=10, M=10, N=0, surface=surf2, pressure=pressure, iota=iota, Psi=1.0)
eq1, info = eq.solve(verbose=3, copy=True, ftol=1e-6)


# plot_comparison(
#     [eq, eq1,], labels=["Initial", "eq.solve"]
# );

L, M, N = 8, 8, 0
lg = desc.grid.QuadratureGrid(L, M, N, NFP=eq.NFP)
data = eq1.compute(["B", "rho", "psi", "Z"], grid=lg)

Br, Bphi, Bz = data["B"].T
dV = lg.weights * data["sqrt(g)"]
pts = np.array([data["R"], data["Z"], data["rho"], Br, Bphi, Bz, dV]).T

np.savetxt(
    "desc_elliptical_equilibrium_b_field.csv",
    pts,
    delimiter=",",
    header="R, Z, rho, Br, Bphi, Bz, dV",
    fmt="%14.8f",
)

# check that volume sums correctly
R0 = 0.2
a0 = 0.05
kappa = 1.5
pi = np.pi

V = 2 * pi * R0 * pi * a0**2 * kappa
print(f"Exact ellipse volume is {V=:.6f} m^3")
print(f"Computed volume is {sum(lg.weights * data['sqrt(g)']):.6g}")
