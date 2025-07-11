import jax
from jax.lax import rsqrt
from jax.numpy import concatenate, diff, zeros_like, linspace, trapezoid
from scipy.constants import c

c = c/1000


def kahan_cumsum(x):
    """Kahan summation algorithm for improved numerical precision."""
    def body_fun(carry, xi):
        s, c = carry
        y = xi - c
        t = s + y
        c = (t - s) - y
        return (t, c), t

    init_carry = (x[0], 0.0)
    _, result = jax.lax.scan(body_fun, init_carry, x[1:])
    return concatenate([x[:1], result])


@jax.jit
def cumulative_trapezoid(y, x):
    dx = diff(x, axis=-1)
    t = (y[..., :-1] + y[..., 1:]) / 2 * dx
    return concatenate([zeros_like(y[..., :1]), kahan_cumsum(t)], axis=-1)


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
