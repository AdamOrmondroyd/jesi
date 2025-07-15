import jax
from jax.lax import rsqrt
from jax.numpy import concatenate, diff, zeros_like, linspace, trapezoid
from scipy.constants import c
from functools import partial, wraps

c = c/1000


@wraps(partial)
def partial_with_requirements(f, *args, **kwargs):
    requirements = f.requirements.copy()
    f = partial(f, *args, **kwargs)
    f.requirements = requirements
    return f


@jax.jit
def cumulative_trapezoid(y, x):
    dx = diff(x, axis=-1)
    t = (y[..., :-1] + y[..., 1:]) / 2 * dx

    # Test: replace cumsum with scan to see if this fixes CPU/GPU differences
    def scan_fn(carry, x):
        return carry + x, carry + x
    _, cumulative = jax.lax.scan(scan_fn, 0.0, t)
    return concatenate([zeros_like(y[..., :1]), cumulative], axis=-1)


def one_over_h(f_de, z, params):
    """
    1/(H(z) / H0)

    f_de(z, *args, **kwargs)
    """
    return rsqrt(
        params['omegam'] * (1 + z)**3
        + (1 - params['omegam']) * f_de(z, params)
    )


def int_one_over_h(one_over_h, z, params, resolution=1000):
    """
    Integrate 1/(H(z) / H0) from 0 to z.
    """
    _z = linspace(0, z, resolution, axis=0)
    # _one_over_h = [one_over_h(__z, omegam, *args, **kwargs) for __z in _z]
    _one_over_h = one_over_h(_z, params)
    return trapezoid(_one_over_h, _z, axis=0)


def dh_over_rs(one_over_h, z, params):
    return c / params['h0rd'] * one_over_h(z, params)


def dm_over_rs(int_one_over_h, z, params, resolution=1000):
    return c / params['h0rd'] * int_one_over_h(z, params,
                                               resolution=resolution)


def dv_over_rs(dh_over_rs, dm_over_rs, z, params):
    return (z * dm_over_rs(z, params) ** 2
            * dh_over_rs(z, params)) ** (1/3)


def h0_dl_over_c(one_over_h, int_one_over_h, zhd, zhel, params):
    """
    H0 * D_L / c
    """
    q0 = int_one_over_h(zhd[0], params)
    h_inverse = one_over_h(zhd, params)
    q = cumulative_trapezoid(h_inverse, zhd) + q0
    return (1 + zhel) * q


for f in (one_over_h, int_one_over_h, h0_dl_over_c):
    f.requirements = {'omegam'}

for f in dh_over_rs, dm_over_rs, dv_over_rs:
    f.requirements = {'omegam', 'h0rd'}
