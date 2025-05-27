import numpy as np
import jax.numpy as jnp
from pandas import read_csv
import cosmology
import jax

omegar = 8.24e-5

data = read_csv("../distances/desi3/desi3_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt("../distances/desi3/desi3_cov.txt")

zs = jnp.array(data.iloc[:, 0].to_numpy())
mean = data.iloc[:, 1].to_numpy()
di_over_rss = [getattr(cosmology, i.lower()) for i in data.iloc[:, 2]]

invcov_over_2 = jnp.linalg.inv(cov) / 2
lognorm = -jnp.linalg.slogdet(2*jnp.pi*cov)[1] / 2


@jax.jit
def logl_desidr2(params):
    h0rd = params['h0rd']
    omegam = params['omegam']
    w0 = params['w0']
    wa = params['wa']

    x = jnp.array([
        di_over_rs(z, w0, wa, h0rd, omegam, omegar)
        for z, di_over_rs in zip(zs, di_over_rss)
    ]).squeeze()

    y = x - mean
    new = -y[None, :] @ invcov_over_2 @ y[:, None] + lognorm
    return new.squeeze()
