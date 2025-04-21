import jax.numpy as jnp
from jax.lax import scan

sqrt = jnp.sqrt

NUM_LOOPS = 9
def rf(x, y, z):
    """
    """
   
    xyz = jnp.array([x, y, z])
    A0 = jnp.sum(xyz) / 3
    init = A0, xyz
    
    def body(carry, _):
        An, xyz = carry
        λ = sqrt(xyz[0]*xyz[1]) + sqrt(xyz[0]*xyz[2]) + sqrt(xyz[1]*xyz[2])

        An_new = (An + λ) / 4
        xyz_new = (xyz + λ) / 4

        return (An_new, xyz_new), None

    result, _ = scan(body, init, length=NUM_LOOPS)
    an, _ = result
    
    f = 4**(-NUM_LOOPS)

    x = (A0 - x) / an * f
    y = (A0 - y) / an * f
    z = -(x + y)
    E2 = x * y - z * z
    E3 = x * y * z

    return (
        1 
        - E2 / 10
        + E3 / 14 
        + E2 * E2 / 24 
        - 3 * E2 * E3 / 44
    ) / sqrt(an)

def rd(x, y, z):
    pass
