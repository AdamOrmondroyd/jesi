# %%
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from desidr2 import logl_desidr2
import lcdm
from nested_sampling import nested_sampling
from jax.scipy.special import logsumexp
tfd = tfp.distributions

rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)

h0rd_prior = tfd.Uniform(1000, 100_000)
omegam_prior = tfd.Uniform(0.01, 0.99)

prior = tfd.JointDistributionNamed(dict(
    h0rd=h0rd_prior,
    omegam=omegam_prior,
))

# %%
nlive = 5000

rng_key, init_key = jax.random.split(rng_key, 2)

prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
logl_samples = jax.vmap(lambda x: logl_desidr2(x, lcdm))(prior_samples)

# %%
labels = [("h0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}")]

samples = nested_sampling(
    lambda x: logl_desidr2(x, lcdm), prior.log_prob, nlive, rng_key, "chains/lcdm",
    labels, prior_samples, logl_samples)

print(f"anesthetic logZ = {samples.logZ():.2f} = {samples.logL_P():.2f} - {samples.D_KL():.2f}")

# %%
nmonaco = 100_000
monaco = logsumexp(
    jax.vmap(lambda x: logl_desidr2(x, lcdm))(prior.sample(seed=rng_key, sample_shape=(nmonaco,)))
) - jnp.log(nmonaco)
print(f"Monte Carlo logZ = {monaco:.2f}")
