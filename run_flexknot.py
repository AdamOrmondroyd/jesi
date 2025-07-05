import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
import sys
import jax
from desidr2 import logl_desidr2
from pantheonplus import logl_pantheonplus
import flexknot
from nested_sampling import sample_flexknot

n = int(sys.argv[1])
assert n >= 2, "n must be at least 2"

rng_key = jax.random.PRNGKey(60022)
os.makedirs("chains", exist_ok=True)


def logl(x):
    return logl_desidr2(x, flexknot) + logl_pantheonplus(x, flexknot)


samples = sample_flexknot(logl, 50, f"chains/dp_flexknot_{n}", rng_key, n=n)
print(f"anesthetic logZ = {samples.logZ():.2f} = {samples.logL_P():.2f} - {samples.D_KL():.2f}")
