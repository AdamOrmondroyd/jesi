from jax.numpy import (
    exp,
    log,
    inf,
    ones,
    zeros,
    concatenate,
    argmin,
    where,
)
from jayesian.cosmology.cosmology import (
    one_over_h,
    int_one_over_h,
    dh_over_rs,
    dm_over_rs,
    dv_over_rs,
    h0_dl_over_c,
    partial_with_requirements as partial,
)


def integrate_cpl(ai, ai1, wi, wi1, alower=None):
    m = (wi - wi1) / (ai - ai1)
    oneplusc = wi - m * ai + 1
    if alower is None:
        alower = ai1
    return oneplusc * log(ai/alower) + m * (ai - alower)


def sections(a, w):
    # compute all but the last part of ∫1+w/1+zdz, and preemtively cumsum
    # Never need the last section precomputed as would need z=infinity
    # a, w are 1D arrays
    a = concatenate([ones(1, dtype=a.dtype), a, zeros(1, dtype=a.dtype)])
    sections = integrate_cpl(a[:-2], a[1:-1], w[:-2], w[1:-1])
    # helpful to have zero at the beginning to handle
    # being in the zeroth flexknot section
    sections = sections.cumsum(axis=0)
    sections = concatenate([zeros(1), sections])
    return a, sections


def f_de(z, params):
    a, w = params['a'], params['w']
    a, _sections = sections(a, w)
    alower = 1/(1+z)
    i = argmin(where(a > alower[..., None], a, inf), axis=-1)
    ai = a[i]
    ai1 = a[i+1]
    wi = w[i]
    wi1 = w[i+1]
    section = _sections[i]
    return exp(3*(section + integrate_cpl(ai, ai1, wi, wi1, alower)))


one_over_h = partial(one_over_h, f_de)
int_one_over_h = partial(int_one_over_h, one_over_h)

dh_over_rs = partial(dh_over_rs, one_over_h)
dm_over_rs = partial(dm_over_rs, int_one_over_h)
dv_over_rs = partial(dv_over_rs, dh_over_rs, dm_over_rs)

h0_dl_over_c = partial(h0_dl_over_c, one_over_h, int_one_over_h)

for f in (one_over_h, int_one_over_h, dh_over_rs, dm_over_rs, dv_over_rs, h0_dl_over_c):
    f.requirements |= {'a', 'w'}
