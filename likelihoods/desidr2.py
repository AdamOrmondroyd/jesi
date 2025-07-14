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


def logl(params, cosmology):
    x = [
        di_over_rs(z, params)
        for z, di_over_rs in zip(zs, [
            getattr(cosmology, i.lower()) for i in data.iloc[:, 2]
        ])
    ]
    x = jnp.array(x)

    y = x - mean
    new = -y[None, :] @ invcov_over_2 @ y[:, None] + lognorm
    return new.squeeze()


# Assign requirements as function attribute
logl.requirements = set(i.lower() for i in data.iloc[:, 2])
