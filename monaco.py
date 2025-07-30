import os
if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config
    config.update("jax_enable_x64", False)

import jax
import numpy as np
from jayesian.likelihoods import pantheonplus, des5y
from jayesian.cosmology import lcdm


nsamples = 10000
omegam = {"omegam": jax.numpy.array(np.random.rand(nsamples)*(0.98)+0.01)}
print(omegam)

logl = jax.vmap(lambda om: pantheonplus(om, lcdm))(omegam)
print(jax.scipy.special.logsumexp(logl) - np.log(nsamples))
