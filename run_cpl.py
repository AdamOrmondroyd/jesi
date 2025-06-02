# %%
import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from desidr2 import logl_desidr2
import cpl
from nested_sampling import nested_sampling
from jax.scipy.special import logsumexp
tfd = tfp.distributions

rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)

h0rd_prior = tfd.Uniform(1000, 100_000)  # 1/99_000
omegam_prior = tfd.Uniform(0.01, 0.99)  # 1/0.98
w0_prior = tfd.Uniform(-3.0, 1.0)  # 1/4
wa_prior = tfd.Uniform(-3.0, 2.0)  # 1/5

cuboid_prior = tfd.JointDistributionNamed(dict(
    h0rd=h0rd_prior,
    omegam=omegam_prior,
    w0=w0_prior,
    wa=wa_prior,
))


logv = jnp.log((20-9/2) * (0.99-0.01) * (100_000 - 1_000))


def log_prob(x):
    # return -logv if within the prior bounds, else -inf
    return jnp.where(x['w0'] + x['wa'] < 0,
                     cuboid_prior.log_prob(x) + jnp.log(20/(20-9/2)),
                     -jnp.inf)


# NOTE: do not use this for the prior probabilities, as normalization is fucked
prior_sample_generator = tfd.JointDistributionNamed(dict(
    h0rd=h0rd_prior,
    omegam=omegam_prior,
    w0=w0_prior,
    wa=lambda w0: tfd.Uniform(-3.0, jnp.minimum(2.0, -w0))
))

# %%
nlive = 5000

rng_key, init_key = jax.random.split(rng_key, 2)

prior_samples = prior_sample_generator.sample(seed=init_key, sample_shape=(nlive,))
logl_samples = jax.vmap(lambda x: logl_desidr2(x, cpl))(prior_samples)

# %%
labels = [("h0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}"),
          ("w0", r"w_0"), ("wa", r"w_a")]

samples = nested_sampling(lambda x: logl_desidr2(x, cpl), log_prob, nlive,
                          rng_key, "chains/cpl", labels, prior_samples,
                          logl_samples)

print(f"anesthetic logZ = {samples.logZ():.2f} = {samples.logL_P():.2f} - {samples.D_KL():.2f}")

# %%
nmonaco = 100_000
monaco = logsumexp(
    jax.vmap(lambda x: logl_desidr2(x, cpl))(prior_sample_generator.sample(seed=rng_key, sample_shape=(nmonaco,)))
) - jnp.log(nmonaco)
print(f"Monte Carlo logZ = {monaco:.2f}")
