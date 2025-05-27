import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import numpy as np
import jax
import jax.numpy as jnp
import blackjax
import tensorflow_probability.substrates.jax as tfp
from tqdm import tqdm
import anesthetic
from desidr2 import logl_desidr2
import cpl
from blackjax.ns.utils import finalise
from jax.scipy.special import logsumexp


def nested_sampling(log_likelihood, log_prior, nlive, rng_key, filename,
                    labels, prior_samples, logl_samples,
                    ravel_fn=None,
                    ):
    n_delete = nlive // 2
    dead = []

    ns = blackjax.ns.adaptive.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        n_delete=n_delete,
        num_mcmc_steps=4*3,
        ravel_fn=ravel_fn,
    )

    def integrate(ns, rng_key):
        rng_key, init_key = jax.random.split(rng_key, 2)

        state = ns.init(prior_samples, logl_samples)

        @jax.jit
        def one_step(carry, xs):
            state, k = carry
            k, subk = jax.random.split(k, 2)
            state, dead_point = ns.step(subk, state)
            return (state, k), dead_point

        one_step((state, rng_key), None)
        with tqdm(desc="Dead points", unit=" dead points") as pbar:
            while (not state.sampler_state.logZ_live - state.sampler_state.logZ < -3):
                (state, rng_key), dead_info = one_step((state, rng_key), None)
                dead.append(dead_info)
                pbar.update(n_delete)

        return state, finalise(state, dead)

    state, final = integrate(ns, rng_key)
    print(f"sampler logZ = {state.sampler_state.logZ:.2f}")

    theta = np.vstack([*final.particles.values()]).T

    labels_map = {l[0]: f'${l[1]}$' for l in labels}

    samples = anesthetic.NestedSamples(
        data=theta,
        logL=final.logL,
        logL_birth=final.logL_birth,
        columns=[l[0] for l in labels],
        labels=labels_map,
    )
    samples.to_csv(f"{filename}.csv")
    return samples
