import jax.numpy as jnp

from anarrima.ray_cone import cosine_of_limiting_angle, r_contact, t_contact


def _segment_is_outward_facing(r, z):
    # Get consecutive pairs for segments
    z_closed = jnp.append(z, 0)
    z1, z2 = z_closed[:-1], z_closed[1:]

    # Outward-facing segments have z2 < z1
    is_outward = z2 < z1

    return is_outward


def wall_point(r, z, i, t):
    """
    Args:
        r: set of radii of wall vertices
        i: index of wall
        t: lerp parameter along segment [0, 1]
    """
    r0, r1, z0, z1 = _p0p1(r, z)
    r_wall = r0[i] * (1 - t) + r1[i] * t
    z_wall = z0[i] * (1 - t) + z1[i] * t
    return r_wall, z_wall


def _p0p1(r, z):
    r0 = r
    r1 = jnp.roll(r, -1)
    z0 = z
    z1 = jnp.roll(z, -1)
    return r0, r1, z0, z1


def _get_ms(r, z):
    # might need a trick here to avoid nans in gradients for flat bits?
    r0, r1, z0, z1 = _p0p1(r, z)

    rdiff = r1 - r0  # rise
    zdiff = z1 - z0  # run
    return rdiff / zdiff


def _x_of_segments_at_height_of_w(r0, z0, m, zw):
    x = r0 - m * (z0 - zw)
    return x


def visible_angle(r, z, i, t, rp, zp):
    m = _get_ms(r, z)
    finite_m = jnp.isfinite(m)
    facing_outward = _segment_is_outward_facing(r, z)

    rw, zw = wall_point(r, z, i, t)

    # z_rel = z - zw # relative to wall point
    zp_rel = zp - zw
    x = _x_of_segments_at_height_of_w(r, z, m, zw)

    # ray_cone functions need w=rw, p=rp, zp=zp_rel, x=x, m=m
    r_c = r_contact(w=rw, p=rp, z=zp_rel, x=x, m=m)
    t_c = t_contact(w=rw, p=rp, z=zp_rel, x=x, m=m)
    u = cosine_of_limiting_angle(w=rw, p=rp, z=zp_rel, x=x, m=m)

    # print(u[i] == u_self[i])
    # u = u.at[i].set(u_self[i])

    t_c_okay = (t_c >= 0) & (t_c <= 1)  # contact is along the ray
    positive_r_c = r_c >= 0
    r_c_finite = jnp.isfinite(r_c)
    u_finite = jnp.isfinite(u)

    testable_segment = facing_outward & finite_m
    valid_segment = testable_segment & (positive_r_c & u_finite & r_c_finite & t_c_okay)
    u_eff = jnp.where(valid_segment, u, -1.0)
    return jnp.max(u_eff)
