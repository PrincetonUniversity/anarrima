import jax.numpy as jnp
from pytest import approx

import anarrima.reduced_integrals as g

π = jnp.pi

# default values
p = 1.0
z = 2.0
r = 0.5
f = φ = 0.6

α = 0.3
β = 1.3

high_accuracy = 4e-16


# Basic tests of isotropic mode
def test_g_HI():
    g.g_HI(p=p, z=z, r=r, φ=f)


def test_g_HI_0():
    ans = g.g_HI(p=p, z=z, r=r, φ=0)
    assert ans == 0


def test_g_HI_pi():
    ans = g.g_HI(p=p, z=z, r=r, φ=π)


def test_g_HI_z_symmetry():
    ans_1 = g.g_HI(p=p, z=z, r=r, φ=f)
    ans_2 = g.g_HI(p=p, z=-z, r=r, φ=f)
    assert ans_1 == ans_2


def test_g_HI_same_height():
    ans_1 = g.g_HI(p=p, z=0, r=r, φ=f)
    ans_2 = g.g_HI_same_height(p=p, r=r, φ=f)
    assert ans_1 == ans_2


# See what accuracy is possible
def test_g_HA():
    ans = g.g_HA(p=p, z=z, r=r, φ=f)
    mathematica_answer = 0.058981191793359559063
    assert ans == approx(mathematica_answer, abs=3e-16)


# test that sums of A and cos equal the isotropic distributions
def test_g_HI_sum():
    iso = g.g_HI(p=p, z=z, r=r, φ=f)
    a = g.g_HA(p=p, z=z, r=r, φ=f)
    c = g.g_Hc(p=p, z=z, r=r, φ=f)
    assert iso == approx(a + c, abs=high_accuracy)


# test relationship between g_HI, g_HBa and g_Hca
def test_g_HB_sum():
    b = g.g_HBa(p=p, z=z, r=r, α=α, β=β, φ=f)
    iso = g.g_HI(p=p, z=z, r=r, φ=f)
    c = g.g_Hca(p=p, z=z, r=r, α=α, β=β, φ=f)
    assert b == approx(iso / 4 + 3 * c / 4, abs=high_accuracy)


# Angled versus toroidal
def test_g_HA_vs_g_HAa():
    toroidal = g.g_HA(p=p, z=z, r=r, φ=f)
    angled = g.g_HAa(p=p, z=z, r=r, α=0, β=0, φ=f)
    assert toroidal == approx(angled, rel=high_accuracy)


def test_g_Hc_vs_g_Hca():
    toroidal = g.g_Hc(p=p, z=z, r=r, φ=f)
    angled = g.g_Hca(p=p, z=z, r=r, α=0, β=0, φ=f)
    assert toroidal == approx(angled, rel=high_accuracy)


def test_g_HB_vs_g_HBa():
    toroidal = g.g_HB(p=p, z=z, r=r, φ=f)
    angled = g.g_HBa(p=p, z=z, r=r, α=0, β=0, φ=f)
    assert toroidal == approx(angled, rel=high_accuracy)


# Tests of vertical modes
# -----------------------
def test_g_VI_zero():
    ans = g.g_VI(p=p, z=z, r=r, φ=0)
    assert ans == 0


# Z antisymmetry
def test_g_VI_z_antisymmetry():
    pos = g.g_VI(p=p, z=z, r=r, φ=f)
    neg = g.g_VI(p=p, z=-z, r=r, φ=f)
    assert pos == -neg


def test_g_VI_sum():
    iso = g.g_VI(p=p, z=z, r=r, φ=f)
    a = g.g_VAa(p=p, z=z, r=r, α=α, β=β, φ=f)
    c = g.g_Vca(p=p, z=z, r=r, α=α, β=β, φ=f)
    assert iso == approx(a + c, abs=high_accuracy)


# Relationship between g_VI, g_VBa and g_Vca
def test_g_VB_sum():
    b = g.g_VBa(p=p, z=z, r=r, α=α, β=β, φ=f)
    iso = g.g_VI(p=p, z=z, r=r, φ=f)
    c = g.g_Vca(p=p, z=z, r=r, α=α, β=β, φ=f)
    assert b == approx(iso / 4 + 3 * c / 4, abs=high_accuracy)


# Angled versus toroidal
def test_g_VA_vs_g_VAa():
    toroidal = g.g_VA(p=p, z=z, r=r, φ=f)
    angled = g.g_VAa(p=p, z=z, r=r, α=0, β=0, φ=f)
    assert toroidal == approx(angled, rel=high_accuracy)


def test_g_Vc_vs_g_Vca():
    toroidal = g.g_Vc(p=p, z=z, r=r, φ=f)
    angled = g.g_Vca(p=p, z=z, r=r, α=0, β=0, φ=f)
    assert toroidal == approx(angled, rel=high_accuracy)


def test_g_VB_vs_g_VBa():
    toroidal = g.g_VB(p=p, z=z, r=r, φ=f)
    angled = g.g_VBa(p=p, z=z, r=r, α=0, β=0, φ=f)
    assert toroidal == approx(angled, rel=high_accuracy)
