import jax
from jax.lax import rsqrt
from jax.numpy import linspace, trapezoid, diff, concatenate, zeros_like
from scipy.constants import c

c = c/1000


@jax.jit
def cumulative_trapezoid(y, x):
    dx = diff(x, axis=-1)
    t = (y[..., :-1] + y[..., 1:]) / 2 * dx
    return concatenate([zeros_like(y[..., :1]), t.cumsum(axis=-1)], axis=-1)


def one_over_h(z, omegam):
    """
    1/(H(z) / H0)
    """
    return rsqrt(
        omegam * (1 + z)**3
        + (1 - omegam)
    )


def int_one_over_h(z, omegam, resolution=1000):
    """
    Integrate 1/(H(z) / H0) from 0 to z.
    """
    _z = linspace(0, z, resolution, axis=-1)
    _one_over_h = one_over_h(_z, omegam[..., None])
    return trapezoid(_one_over_h, _z, axis=-1)


@jax.jit
def dh_over_rs(z, h0rd, omegam):
    return c / h0rd * one_over_h(z, omegam)


def dm_over_rs(z, h0rd, omegam, resolution=1000):
    return c / h0rd * int_one_over_h(z, omegam, resolution=resolution)


def dv_over_rs(z, *args, **kwargs):
    return (z * dm_over_rs(z, *args, **kwargs) ** 2
            * dh_over_rs(z, *args, **kwargs)) ** (1/3)


def h0_dl_over_c(zhd, zhel, omegam, **_):
    """
    H0 * D_L / c
    """
    q0 = int_one_over_h(zhd[0], omegam)
    h_inverse = one_over_h(zhd, omegam)
    q = cumulative_trapezoid(h_inverse, zhd) + q0
    return (1 + zhel) * q
