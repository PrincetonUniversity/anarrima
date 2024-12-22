import anarrima.reduced_integrals as g
import numpy as np
from pytest import approx

π = np.pi

# default values
pp = 1.0
zz = 2.0
rr = 0.5
ff = φφ = 0.6

def test_g_HI():
    g.g_HI(p=pp, z=zz, r=rr, φ=ff)

def test_g_HI_0():
    ans = g.g_HI(p=pp, z=zz, r=rr, φ=0)
    assert ans == 0

def test_g_HI_pi():
    ans = g.g_HI(p=pp, z=zz, r=rr, φ=π)

def test_g_HI_symmetry():
    ans_1 = g.g_HI(p=pp, z=zz, r=rr, φ=ff)
    ans_2 = g.g_HI(p=pp, z=-zz, r=rr, φ=ff)
    assert ans_1 == ans_2

def test_g_HI_same_height():
    ans_1 = g.g_HI(p=pp, z=0, r=rr, φ=ff)
    ans_2 = g.g_HI_same_height(p=pp, r=rr, φ=ff)
    assert ans_1 == ans_2

def test_g_HA():
    ans = g.g_HA(p=pp, z=zz, r=rr, φ=ff)
    mathematica_answer = 0.058981191793359559063
    assert ans == approx(mathematica_answer, abs=1e-16)
