from pathlib import Path
import numpy as np
from jax import vmap
from jax.lax import switch
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

# Function name to index mapping
func_map = {"dv_over_rs": 0, "dm_over_rs": 1, "dh_over_rs": 2}
func_names = data.iloc[:, 2].str.lower().tolist()
func_indices = jnp.array([func_map[name] for name in func_names])


def compute_single(z, func_idx, params, cosmology):
    return switch(
        func_idx,
        [cosmology.dv_over_rs, cosmology.dm_over_rs, cosmology.dh_over_rs],
        z, params
    )


compute_vectorized = vmap(compute_single, in_axes=(0, 0, None, None))


def logl(params, cosmology):
    x = compute_vectorized(zs, func_indices, params, cosmology)

    y = x - mean
    new = -y[None, :] @ invcov_over_2 @ y[:, None] + lognorm
    return new.squeeze()


# Assign requirements as function attribute
logl.requirements = set(i.lower() for i in data.iloc[:, 2])
