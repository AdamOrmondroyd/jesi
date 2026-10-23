import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
import jax
import anesthetic
from tensorflow_probability.substrates.jax import distributions as tfd
import blackjax
from blackjax.ns.utils import finalise
from jax.scipy.stats import norm
from jax import numpy as jnp
from jax.scipy.special import logsumexp
from tqdm import tqdm


rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)

x_prior = tfd.Uniform(-1.0, 1.0)
y_prior = tfd.Uniform(-1.0, 1.0)

prior = tfd.JointDistributionNamed(dict(
    x=x_prior,
    y=y_prior,
))

nDims = 2
sigma = 0.05
centres = jnp.array([[-0.5, -0.5], [0.5, 0.5]])

weights = jnp.array([0.5, 0.5])



# Correctly normalised Gaussian mixture with two peaks at +/-(0.5, 0.5)
def logl(theta):
    theta_vec = jnp.array([theta['x'], theta['y']])
    
    # Compute log probability for each Gaussian component
    log_prob_1 = jnp.sum(norm.logpdf(theta_vec, loc=centres[0], scale=sigma))
    log_prob_2 = jnp.sum(norm.logpdf(theta_vec, loc=centres[1], scale=sigma))
    
    # Weighted mixture using logsumexp for numerical stability
    log_weights = jnp.log(weights)
    log_probs = jnp.array([log_prob_1, log_prob_2])
    
    return logsumexp(log_weights + log_probs)

nlive = 500

rng_key, init_key = jax.random.split(rng_key, 2)

prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
logl_samples = jax.vmap(logl)(prior_samples)


def nested_sampling(log_likelihood, log_prior, nlive, rng_key,
                    prior_samples, logl_samples,
                    ):
    n_delete = nlive // 2
    dead = []

    ns = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=8,
    )

    def integrate(ns, rng_key):
        state = ns.init(prior_samples)

        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = ns.step(subk, state)
            return (state, k), dead_point

        one_step((state, rng_key), None)
        with tqdm(desc="Dead points", unit=" dead points") as pbar:
            while (not state.logZ_live - state.logZ < -3):
                (state, rng_key), dead_info = one_step((state, rng_key), None)
                dead.append(dead_info)
                pbar.update(n_delete)

        return state, finalise(state, dead)

    state, final = integrate(ns, rng_key)
    print(f"sampler logZ = {state.logZ:.2f}")

    return anesthetic.NestedSamples(
        data=final.particles,
        logL=final.loglikelihood,
        logL_birth=final.loglikelihood_birth,
        columns=['x', 'y'],
    )


for i in tqdm(range(100)):
    rng_key, init_key = jax.random.split(rng_key, 2)
    samples = nested_sampling(
        logl, prior.log_prob, nlive, init_key,
        prior_samples, logl_samples)
    samples.to_csv(f"chains/double/double_{i}")
