from jax.numpy import (
    exp,
    log,
    ones,
    zeros,
    concatenate,
    clip,
)
from jesi.cosmology.cosmology import (
    one_over_h,
    int_one_over_h,
    dh_over_rs,
    dm_over_rs,
    dv_over_rs,
    h0_dl_over_c,
    requires,
)
from jesi.requirements import partial


def integrate_cpl(ai, ai1, wi, wi1, alower=None):
    m = (wi - wi1) / (ai - ai1)
    oneplusc = wi - m * ai + 1
    if alower is None:
        alower = ai1
    return oneplusc * log(ai/alower) + m * (ai - alower)


@requires('a', 'w')
def f_de(z, params):
    a = concatenate([ones(1), params['a'], zeros(1)])
    w = params['w']
    alower = 1/(1+z)
    lo = clip(alower[..., None], a[1:], a[:-1])
    return exp(3*integrate_cpl(a[:-1], a[1:], w[:-1], w[1:], lo).sum(axis=-1))


one_over_h = partial(one_over_h, f_de)
int_one_over_h = partial(int_one_over_h, one_over_h)

dh_over_rs = partial(dh_over_rs, one_over_h)
dm_over_rs = partial(dm_over_rs, int_one_over_h)
dv_over_rs = partial(dv_over_rs, dh_over_rs, dm_over_rs)

h0_dl_over_c = partial(h0_dl_over_c, one_over_h, int_one_over_h)
