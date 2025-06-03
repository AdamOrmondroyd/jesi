import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import numpy as np
import jax
import blackjax
from tqdm import tqdm
import anesthetic
from blackjax.ns.utils import finalise


def nested_sampling(log_likelihood, log_prior, nlive, rng_key, filename,
                    labels, prior_samples, logl_samples,
                    ):
    n_delete = nlive // 2
    dead = []

    ns = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=4*3,
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

    labels_map = {l[0]: f'${l[1]}$' for l in labels}

    samples = anesthetic.NestedSamples(
        data=final.particles,
        logL=final.loglikelihood,
        logL_birth=final.loglikelihood_birth,
        columns=[l[0] for l in labels],
        labels=labels_map,
    )

    samples.to_csv(f"{filename}.csv")
    return samples
