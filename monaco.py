import os
from jax import config
import jax
import numpy as np
from jesi.likelihoods import pantheonplus, des5y
from jesi.cosmology import lcdm

config.update("jax_enable_x64", False)
print(f"JAX in {'64' if config.jax_enable_x64 else '32'}-bit mode")
print(f"JAX platform: {os.environ.get('JAX_PLATFORM_NAME', 'default')}")

nsamples = 10000
omegam = {"omegam": jax.numpy.array(np.random.rand(nsamples)*(0.98)+0.01)}
print(omegam)

logl = jax.vmap(lambda om: pantheonplus(om, lcdm))(omegam)
print(jax.scipy.special.logsumexp(logl) - np.log(nsamples))
