# %%
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import jax
import tensorflow_probability.substrates.jax as tfp
from nested_sampling import nested_sampling
from jax.scipy.stats import norm


tfd = tfp.distributions

rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)

x_prior = tfd.Uniform(0.0, 1.0)

prior = tfd.JointDistributionNamed(dict(
    x=x_prior,
))

def logl(x): return norm.logpdf(x['x'])

nlive = 5000

rng_key, init_key = jax.random.split(rng_key, 2)

prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
logl_samples = jax.vmap(logl)(prior_samples)

# %%
labels = [("x", "x")]

samples = nested_sampling(
    logl, prior.log_prob, nlive, rng_key, "chains/mwe",
    labels, prior_samples, logl_samples)

print(f"anesthetic logZ = {samples.logZ():.2f} = {samples.logL_P():.2f} - {samples.D_KL():.2f}")
