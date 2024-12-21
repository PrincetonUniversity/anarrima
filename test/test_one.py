import anarrima.reduced_integrals as g
import numpy as np

π = np.pi

# default values
pp = 1.0
zz = 2.0
rr = 0.5
ff = φφ = 0.6

def test_gHI():
    g.g_HI(p=pp, z=zz, r=rr, φ=ff)

def test_gHI_0():
    ans = g.g_HI(p=pp, z=zz, r=rr, φ=0)
    assert ans == 0

def test_gHI_pi():
    ans = g.g_HI(p=pp, z=zz, r=rr, φ=π)

def test_gHI_symmetry():
    ans_1 = g.g_HI(p=pp, z=zz, r=rr, φ=ff)
    ans_2 = g.g_HI(p=pp, z=-zz, r=rr, φ=ff)
    assert ans_1 == ans_2
