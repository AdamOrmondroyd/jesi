import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
from sys import argv
import jax
from pathlib import Path
from tensorflow_probability.substrates.jax import distributions as tfd
from nested_sampling import nested_sampling
import cosmology
from likelihoods import pantheonplus as logl

model = getattr(cosmology, argv[1])

if __name__ == "__main__":

    chains = Path("chains")
    chains.mkdir(exist_ok=True, parents=True)
    rng_key = jax.random.PRNGKey(1729)

    omegam_prior = tfd.Uniform(0.01, 0.99)

    prior = tfd.JointDistributionNamed(dict(omegam=omegam_prior))

    nlive = 5000
    rng_key, init_key = jax.random.split(rng_key, 2)
    prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
    logl_values = jax.vmap(lambda x: logl(x, model))(prior_samples)

    nested_sampling(lambda x: logl(x, model), prior.log_prob,
                    logl_values, prior_samples,
                    nlive, chains/f"pantheonplus_{model}",
                    [("omegam", r"\Omega_\mathrm{m}")], rng_key)
