from pathlib import Path
import numpy as np
import jax.numpy as jnp
from pandas import read_csv

path = Path(__file__).parent/'data/desidr2'
data = read_csv(path/"desidr2_mean.txt",
                header=None, index_col=None, sep=r"\s+", comment="#")
cov = np.loadtxt(path/"desidr2_cov.txt")

zs = jnp.array(data.iloc[:, 0].to_numpy())
mean = jnp.array(data.iloc[:, 1].to_numpy())

invcov_over_2 = jnp.linalg.inv(cov) / 2
lognorm = -jnp.linalg.slogdet(2*jnp.pi*cov)[1] / 2

di_over_rs_names = data.iloc[:, 2].str.lower().tolist()


def logl(params, cosmology):
    di_over_rss = [getattr(cosmology, name) for name in di_over_rs_names]

    x = jnp.array([
        method(z, params)
        for z, method in zip(zs, di_over_rss)
    ])

    y = x - mean
    new = -y[None, :] @ invcov_over_2 @ y[:, None] + lognorm
    return new.squeeze()


# Assign requirements as function attribute
logl.requirements = set(i.lower() for i in data.iloc[:, 2])
