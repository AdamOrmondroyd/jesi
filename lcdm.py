import jax
from jax.lax import rsqrt
from jax.numpy import linspace, trapezoid
from scipy.constants import c

c = c/1000


def one_over_h(z, omegam):
    """
    1/(H(z) / H0)
    """
    return rsqrt(
        omegam * (1 + z)**3
        + (1 - omegam)
    )


@jax.jit
def dh_over_rs(z, h0rd, omegam):
    return c / h0rd * one_over_h(z, omegam)


def dm_over_rs(z, h0rd, omegam, resolution=1000):
    _z = linspace(0, z, resolution, axis=-1)
    _one_over_h = one_over_h(_z, omegam[..., None])
    return c / h0rd * trapezoid(_one_over_h, _z, axis=-1)


def dv_over_rs(z, *args, **kwargs):
    return (z * dm_over_rs(z, *args, **kwargs) ** 2
            * dh_over_rs(z, *args, **kwargs)) ** (1/3)
