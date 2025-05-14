import jax
import jax.numpy as jnp
import mpmath as mp
from pytest import approx, mark

from anarrima.visible_angle import visible_angle

# square cross section with minimum radius of 1
square_box_pts = ((1.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0))
square_box_plasma_point = (1.5, 0.5)
square_box_inner_wall_point = (1.0, 0.5)
square_box_outer_wall_point = (2.0, 0.5)


def test_square_box_visible_angle_inner_wall():
    r, z = jnp.array(square_box_pts).T
    i_seg = 3
    t_seg = 0.5
    result = jnp.arccos(visible_angle(r, z, i_seg, t_seg, *square_box_plasma_point))
    expected = jnp.arccos(1 / 1.5)
    assert result == approx(expected, rel=1e-15)


def test_square_box_visible_angle_outer_wall():
    r, z = jnp.array(square_box_pts).T
    i_seg = 1
    t_seg = 0.5
    result = jnp.arccos(visible_angle(r, z, i_seg, t_seg, *square_box_plasma_point))
    expected = float(mp.acos(1 / 1.5) + mp.acos(1.0 / 2))
    assert result == approx(expected, rel=1e-15)


def test_square_box_visible_angle_lower_wall():
    r, z = jnp.array(square_box_pts).T
    i_seg = 0
    t_seg = 0.5
    result = jnp.arccos(visible_angle(r, z, i_seg, t_seg, *square_box_plasma_point))
    expected = float(2 * mp.acos(2 / 3))
    assert result == approx(expected, rel=1e-15)


def test_square_box_visible_angle_upper_wall():
    r, z = jnp.array(square_box_pts).T
    i_seg = 2
    t_seg = 0.5
    result = jnp.arccos(visible_angle(r, z, i_seg, t_seg, *square_box_plasma_point))
    expected = float(2 * mp.acos(2 / 3))
    assert result == approx(expected, rel=1e-15)
