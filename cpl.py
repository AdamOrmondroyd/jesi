import jax
from jax.lax import rsqrt
from jax.numpy import exp, linspace, trapezoid
from scipy.constants import c

c = c/1000


def cpl_f_de(z, w0, wa):
    return (1+z)**(3*(1+w0+wa) * exp(wa * z / (1+z)))


def one_over_h(z, omegam, f_de):
    """
    1/(H(z) / H0)
    """
    return rsqrt(
        omegam * (1 + z)**3
        + (1 - omegam) * f_de
    )


@jax.jit
def dh_over_rs(z, w0, wa, h0rd, omegam):
    _f_de = cpl_f_de(z, w0, wa)
    return c / h0rd * one_over_h(z, omegam, _f_de)


def dm_over_rs(z, w0, wa, h0rd, omegam, resolution=1000):
    _z = linspace(0, z, resolution, axis=-1)
    _f_de = cpl_f_de(_z, w0[..., None], wa[..., None])
    _one_over_h = one_over_h(_z, omegam[..., None], _f_de)
    return c / h0rd * trapezoid(_one_over_h, _z, axis=-1)


def dv_over_rs(z, *args, **kwargs):
    return (z * dm_over_rs(z, *args, **kwargs) ** 2
            * dh_over_rs(z, *args, **kwargs)) ** (1/3)
