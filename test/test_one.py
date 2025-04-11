import anarrima.reduced_integrals as g
import numpy as np
from pytest import approx

π = np.pi

# default values
p = 1.0
z = 2.0
r = 0.5
f = φ = 0.6

α = 0.3
β = 1.3

high_accuracy = 4e-16

def test_g_HI():
    g.g_HI(p=p, z=z, r=r, φ=f)

def test_g_HI_0():
    ans = g.g_HI(p=p, z=z, r=r, φ=0)
    assert ans == 0

def test_g_HI_pi():
    ans = g.g_HI(p=p, z=z, r=r, φ=π)

def test_g_HI_symmetry():
    ans_1 = g.g_HI(p=p, z=z, r=r, φ=f)
    ans_2 = g.g_HI(p=p, z=-z, r=r, φ=f)
    assert ans_1 == ans_2

def test_g_HI_same_height():
    ans_1 = g.g_HI(p=p, z=0, r=r, φ=f)
    ans_2 = g.g_HI_same_height(p=p, r=r, φ=f)
    assert ans_1 == ans_2

def test_g_HA():
    ans = g.g_HA(p=p, z=z, r=r, φ=f)
    mathematica_answer = 0.058981191793359559063
    assert ans == approx(mathematica_answer, abs=high_accuracy)

# test that sums of A and cos equal the isotropic distributions
def test_g_HI_sum():
    iso = g.g_HI(p=p, z=z, r=r, φ=f)
    a = g.g_HA(p=p, z=z, r=r, φ=f)
    c = g.g_Hc(p=p, z=z, r=r, φ=f)
    assert iso == approx(a + c, abs=high_accuracy)

def test_g_VI_sum():
    iso = g.g_VI(p=p, z=z, r=r, φ=f)
    a = g.g_VAa(p=p, z=z, r=r, α=α, β=β, φ=f)
    c = g.g_Vca(p=p, z=z, r=r, α=α, β=β, φ=f)
    assert iso == approx(a + c, abs=high_accuracy)

# test relationship between g_HI, g_HBa and g_Hca
def test_g_HB_sum():
    b = g.g_HBa(p=p, z=z, r=r, α=α, β=β, φ=f)
    iso = g.g_HI(p=p, z=z, r=r, φ=f)
    c = g.g_Hca(p=p, z=z, r=r, α=α, β=β, φ=f)
    assert b == approx(iso/4 + 3*c/4, abs=high_accuracy)

# test relationship between g_VI, g_VBa and g_Vca
def test_g_VB_sum():
    b = g.g_VBa(p=p, z=z, r=r, α=α, β=β, φ=f)
    iso = g.g_VI(p=p, z=z, r=r, φ=f)
    c = g.g_Vca(p=p, z=z, r=r, α=α, β=β, φ=f)
    assert b == approx(iso/4 + 3*c/4, abs=high_accuracy)

