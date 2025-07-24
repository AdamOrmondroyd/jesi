from jax.numpy import exp
from jayesian.cosmology.cosmology import (
    one_over_h,
    int_one_over_h,
    dh_over_rs,
    dm_over_rs,
    dv_over_rs,
    h0_dl_over_c,
    partial_with_requirements as partial,
)


def f_de(z, params):
    w0, wa = params['w0'], params['wa']
    return (1+z)**(3*(1+w0+wa)) * exp(-3*wa*z/(1+z))


one_over_h = partial(one_over_h, f_de)
int_one_over_h = partial(int_one_over_h, one_over_h)

dh_over_rs = partial(dh_over_rs, one_over_h)
dm_over_rs = partial(dm_over_rs, int_one_over_h)
dv_over_rs = partial(dv_over_rs, dh_over_rs, dm_over_rs)

h0_dl_over_c = partial(h0_dl_over_c, one_over_h, int_one_over_h)

for f in (one_over_h, int_one_over_h, dh_over_rs, dm_over_rs, dv_over_rs, h0_dl_over_c):
    f.requirements |= {'w0', 'wa'}
