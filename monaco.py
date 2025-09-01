import os
if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config
config.update("jax_enable_x64", False)
import jax  # noqa: E402
import numpy as np  # noqa: E402
from jayesian.likelihoods import pantheonplus, des5y  # noqa: E402
from jayesian.cosmology import lcdm  # noqa: E402

print(f"JAX in {'64' if config.jax_enable_x64 else '32'}-bit mode")
print(f"JAX platform: {os.environ.get('JAX_PLATFORM_NAME', 'default')}")

nsamples = 10000
omegam = {"omegam": jax.numpy.array(np.random.rand(nsamples)*(0.98)+0.01)}
print(omegam)

logl = jax.vmap(lambda om: pantheonplus(om, lcdm))(omegam)
print(jax.scipy.special.logsumexp(logl) - np.log(nsamples))
