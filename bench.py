"""Time single NSS steps for the sync and async slice kernels.

Each kernel steps from the same mid-run state with the same key, once with
the real likelihood and once with a cheap Gaussian in the same parameters.
If the async per-evaluation overhead persists with the Gaussian, it is fixed
cost in the FSM loop body; if it only appears with the real likelihood, it
comes from how the likelihood compiles inside that body.

    uv run bench.py flexknot desidr2 des5y --n=6
"""
import time
from functools import partial

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from fire import Fire
from jax.flatten_util import ravel_pytree

from jesi import cosmology, likelihoods
from jesi.nested_sampling import async_nss, counted, setup
from run import determine_requirements

KERNELS = {'sync': blackjax.nss, 'async': async_nss}


def gaussian(prior_samples):
    """Narrow Gaussian centred on the prior mean of the same parameters."""
    flat = jax.vmap(lambda x: ravel_pytree(x)[0])(prior_samples)
    mu, sigma = flat.mean(axis=0), 0.05 * flat.std(axis=0)

    def logl(x):
        return -0.5 * jnp.sum(((ravel_pytree(x)[0] - mu) / sigma) ** 2)

    return logl


def benchmark(logl, log_prior, prior_samples, nlive, num_inner_steps,
              burn, repeats, rng_key):
    build = {
        name: partial(kernel, logprior_fn=log_prior, num_delete=nlive // 2,
                      num_inner_steps=num_inner_steps)
        for name, kernel in KERNELS.items()
    }

    # advance with the sync kernel so both kernels start from the same
    # mid-run state rather than the prior
    ns = build['sync'](loglikelihood_fn=logl)
    state = ns.init(prior_samples)
    burn_step = jax.jit(ns.step)
    for _ in range(burn):
        rng_key, subkey = jax.random.split(rng_key)
        state, _ = burn_step(subkey, state)
    _, step_key = jax.random.split(rng_key)

    results = {}
    for name in KERNELS:
        step = jax.jit(build[name](loglikelihood_fn=logl).step)
        step = step.lower(step_key, state).compile()
        jax.block_until_ready(step(step_key, state))  # warm up
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            jax.block_until_ready(step(step_key, state))
            times.append(time.perf_counter() - start)

        calls = []
        counting = jax.jit(
            build[name](loglikelihood_fn=counted(logl, calls)).step
        )
        jax.block_until_ready(counting(step_key, state))
        results[name] = np.median(times), len(calls)
    return results


def main(model_name, *likelihood_names, nlive=1000, burn=10, repeats=5,
         seed=1729, **kwargs):
    model = getattr(cosmology, model_name)
    logls = [getattr(likelihoods, name) for name in likelihood_names]
    requirements = determine_requirements(model, logls)

    def logl(x):
        return sum(logl(x, model) for logl in logls)

    rng_key = jax.random.PRNGKey(seed)
    logl, log_prior, prior_samples, labels, _, rng_key = setup(
        logl, requirements, nlive, rng_key, **kwargs
    )
    num_inner_steps = 3 * len(labels)
    print(f"{len(labels)} dims, nlive={nlive}, num_delete={nlive // 2}, "
          f"num_inner_steps={num_inner_steps}, {burn} burn-in steps, "
          f"median of {repeats}")

    print(f"{'likelihood':<10} {'kernel':<6} {'ms/step':>9} "
          f"{'evals/step':>11} {'ms/eval':>8}")
    for label, fn in (('real', logl), ('gaussian', gaussian(prior_samples))):
        results = benchmark(fn, log_prior, prior_samples, nlive,
                            num_inner_steps, burn, repeats, rng_key)
        for name, (seconds, evals) in results.items():
            print(f"{label:<10} {name:<6} {seconds * 1e3:>9.1f} "
                  f"{evals:>11d} {seconds / evals * 1e3:>8.3f}")


if __name__ == "__main__":
    Fire(main)
