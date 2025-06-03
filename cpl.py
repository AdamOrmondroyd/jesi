import jax
from jax.lax import rsqrt
from jax.numpy import exp, linspace, trapezoid
from scipy.constants import c
from lcdm import cumulative_trapezoid

c = c/1000


def cpl_f_de(z, w0, wa):
    return (1+z)**(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))


def one_over_h(z, omegam, f_de):
    """
    1/(H(z) / H0)
    """
    return rsqrt(
        omegam * (1 + z)**3
        + (1 - omegam) * f_de
    )


def int_one_over_h(z, w0, wa, omegam, resolution=1000):
    """
    Integrate 1/(H(z) / H0) from 0 to z.
    """
    _z = linspace(0, z, resolution, axis=-1)
    _f_de = cpl_f_de(_z, w0[..., None], wa[..., None])
    _one_over_h = one_over_h(_z, omegam[..., None], _f_de)
    return trapezoid(_one_over_h, _z, axis=-1)


@jax.jit
def dh_over_rs(z, w0, wa, h0rd, omegam):
    _f_de = cpl_f_de(z, w0, wa)
    return c / h0rd * one_over_h(z, omegam, _f_de)


def dm_over_rs(z, w0, wa, h0rd, omegam, resolution=1000):
    return c / h0rd * int_one_over_h(z, w0, wa, omegam, resolution=resolution)


def dv_over_rs(z, *args, **kwargs):
    return (z * dm_over_rs(z, *args, **kwargs) ** 2
            * dh_over_rs(z, *args, **kwargs)) ** (1/3)


def h0_dl_over_c(zhd, zhel, w0, wa, omegam, **_):
    """
    H0 * D_L / c
    """
    q0 = int_one_over_h(zhd[0], w0, wa, omegam)
    h_inverse = one_over_h(zhd, omegam, cpl_f_de(zhd, w0, wa))
    q = cumulative_trapezoid(h_inverse, zhd) + q0
    return (1 + zhel) * q
