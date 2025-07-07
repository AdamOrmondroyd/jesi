import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
import jax
from desidr2 import logl_desidr2
from pantheonplus import logl_pantheonplus
import wcdm
from nested_sampling import sample_wcdm

rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)


def logl(x):
    return logl_desidr2(x, wcdm) + logl_pantheonplus(x, wcdm)


samples = sample_wcdm(logl, 500, "chains/dp_wcdm", rng_key)

print(f"anesthetic logZ = {samples.logZ():.2f} = {samples.logL_P():.2f} - {samples.D_KL():.2f}")
