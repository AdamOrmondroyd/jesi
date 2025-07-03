import jax
from jax.lax import rsqrt
from jax.numpy import concatenate, diff, zeros_like, linspace, trapezoid
from scipy.constants import c

c = c/1000


@jax.jit
def cumulative_trapezoid(y, x):
    dx = diff(x, axis=-1)
    t = (y[..., :-1] + y[..., 1:]) / 2 * dx
    return concatenate([zeros_like(y[..., :1]), t.cumsum(axis=-1)], axis=-1)


def one_over_h(f_de, z, omegam, *args, **kwargs):
    """
    1/(H(z) / H0)

    f_de(z, *args, **kwargs)
    """
    return rsqrt(
        omegam * (1 + z)**3
        + (1 - omegam) * f_de(z, *args, **kwargs)
    )


def int_one_over_h(one_over_h, z, omegam, *args, resolution=1000, **kwargs):
    """
    Integrate 1/(H(z) / H0) from 0 to z.
    """
    _z = linspace(0, z, resolution, axis=-1)
    _one_over_h = one_over_h(_z, omegam[..., None], *args, **kwargs)
    return trapezoid(_one_over_h, _z, axis=-1)


def dh_over_rs(one_over_h, z, h0rd, omegam, *args, **kwargs):
    return c / h0rd * one_over_h(z, omegam, *args, **kwargs)


def dm_over_rs(int_one_over_h, z, h0rd, omegam, *args,
               resolution=1000, **kwargs):
    return c / h0rd * int_one_over_h(z, omegam, *args,
                                     resolution=resolution, **kwargs)


def dv_over_rs(dh_over_rs, dm_over_rs, z, *args, **kwargs):
    return (z * dm_over_rs(z, *args, **kwargs) ** 2
            * dh_over_rs(z, *args, **kwargs)) ** (1/3)


def h0_dl_over_c(one_over_h, int_one_over_h, zhd, zhel, omegam,
                 h0rd, *args, **kwargs):
    """
    H0 * D_L / c
    """
    q0 = int_one_over_h(zhd[0], omegam, *args, **kwargs)
    h_inverse = one_over_h(zhd, omegam, *args, **kwargs)
    q = cumulative_trapezoid(h_inverse, zhd) + q0
    return (1 + zhel) * q
