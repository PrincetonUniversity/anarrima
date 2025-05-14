import jax.numpy as jnp

sqrt = jnp.sqrt

SAFE_SQUARE_ROOT_VAL = 1.0


def r_contact(w, p, z, x, m):
    """Radius of tangency for potentially-blocking wall segment

    Args:
      w: arraylike, real-valued. Radius of wall point.
      p: arraylike, real-valued. Radius of source ring.
      z: arraylike, real-valued. Relative height of source ring.
      m: arraylike, real-valued. Slope of potentially-limiting wall element.
      x: arraylike, real-valued. Radius of the element at wall point height.

    Returns:
      array of tangency radii

    This can be used for a test: a negative value indicates
    that the whole ring is visible. In this case the value from u1 should be
    ignored, as it represents something like the last visible angle were the segment
    to be mirrored through the origin, making a "double cone" shape.
    """

    ztc = t_contact(w, p, z, x, m) * z
    return x + ztc * m


# ray-cone intersections
def t_contact(w, p, z, x, m):
    """Parameterized contact time along the ray from wall point to source ring

    0: at wall point. 1: at source ring.
    The radicand can be negative if the source ring
    or wall point are "inside" the segment.
    It can also be negative if the source point is above the reflection of the
    blocking segment through the origin. Then t will be complex (nan).

    Args:
      w: arraylike, real-valued. Radius of wall point.
      p: arraylike, real-valued. Radius of source ring.
      z: arraylike, real-valued. Relative height of source ring.
      m: arraylike, real-valued. Slope of potentially-limiting wall element.
      x: arraylike, real-valued. Radius of the element at wall point height.

    Returns:
      array of cone frustum contact parameters
    """
    xp = x + m * z
    x2, w2, p2, xp2 = w**2, x**2, p**2, xp**2
    radicand = (w2 - x2) * (p2 - xp2)
    test = (radicand > 0) & jnp.isfinite(radicand)
    radicand_fixed = jnp.where(test, radicand, SAFE_SQUARE_ROOT_VAL)
    root = jnp.where(test, sqrt(radicand_fixed), 0.0)
    numer = -x2 + w2 - root
    denom = -p2 + w2 + (xp2 - x2)
    t = numer / denom
    return t


def cosine_of_limiting_angle_self(w, p, z, _, m):
    """Cosine of maximum visible source ring angle

    Args:
      w: arraylike, real-valued. Radius of wall point.
      p: arraylike, real-valued. Radius of source ring.
      z: arraylike, real-valued. Relative height of source ring.
      _: where x would go.
      m: arraylike, real-valued. Slope of potentially-limiting wall element.

             x(x + m z)
    cos φm = ----------
                p w

    """
    return (w + m * z) / p


def cosine_of_limiting_angle(w, p, z, x, m):
    """Cosine of maximum visible source ring angle

    Args:
      w: arraylike, real-valued. Radius of wall point.
      p: arraylike, real-valued. Radius of source ring.
      z: arraylike, real-valued. Rel. height of source ring to wall point.
      m: arraylike, real-valued. Slope of potentially-limiting wall element.
      x: arraylike, real-valued. Radius of the element at wall point height.

             x(x + m z) - √((w² - x²)(p² - (x + m z)²))
    cos φm = --------------------------------------------
                                p w

    """
    xp = x + m * z  # future: avoid ∞ * 0 = 0?
    radicand = (w**2 - x**2) * (p**2 - xp**2)
    # I don't understand quite why I need this
    test = radicand > 0 & jnp.isfinite(radicand)
    radicand_fixed = jnp.where(test, radicand, SAFE_SQUARE_ROOT_VAL)
    root = jnp.where(test, sqrt(radicand_fixed), 0.0)
    numer = x * xp - root
    denom = p * w
    u1 = numer / denom
    # secondary gutter
    # u1_fixed = jnp.where(radicand < 0, -1.0, u1)
    return u1


# algorithm:
# get all segments other than the one the test wall point is on.
# get only segments that span between the wall point and the source ring
# subtract z coordinates so that wall is zero height
# for each segment: get u1 and r1
# if u1 or r1 is nan or if r1 < 0, throw it out (or set u1 to -1)
# so now we have a list of all positive r1's.
# (for a concave cross section, if r1 is negative (not limiting) there will always be another segment which is limiting)
# get the largest u1: this is the most limiting segment
