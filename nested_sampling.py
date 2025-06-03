import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
from jax import numpy as jnp
import jax
import blackjax
from tqdm import tqdm
import anesthetic
from blackjax.ns.utils import finalise
from tensorflow_probability.substrates.jax import distributions as tfd


h0rd_prior = tfd.Uniform(1_000.0, 100_000.0)  # 1/99_000
omegam_prior = tfd.Uniform(0.01, 0.99)  # 1/0.98
w0_prior = tfd.Uniform(-3.0, 1.0)  # 1/4
wa_prior = tfd.Uniform(-3.0, 2.0)  # 1/5


def nested_sampling(log_likelihood, log_prior, prior_samples, logl_samples,
                    nlive, filename, labels, rng_key,
                    ):
    n_delete = nlive // 2
    dead = []

    ns = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=3*len(labels),
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


def sample_lcdm(logl, nlive, filename, rng_key):
    prior = tfd.JointDistributionNamed(dict(
        h0rd=h0rd_prior,
        omegam=omegam_prior,
    ))
    rng_key, init_key = jax.random.split(rng_key, 2)

    prior_samples = prior.sample(seed=init_key, sample_shape=(2*nlive,))
    logl_samples = jax.vmap(logl)(prior_samples)

    labels = [("h0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}")]

    return nested_sampling(
        logl, prior.log_prob, logl_samples,
        prior_samples, nlive,
        filename, labels, rng_key)


def sample_cpl(logl, nlive, filename, rng_key):
    cuboid_prior = tfd.JointDistributionNamed(dict(
        h0rd=h0rd_prior,
        omegam=omegam_prior,
        w0=w0_prior,
        wa=wa_prior,
    ))

    def log_prior(x):
        # return -logv if within the prior bounds, else -inf
        return jnp.where(x['w0'] + x['wa'] < 0,
                         cuboid_prior.log_prob(x) + jnp.log(20/(20-9/2)),
                         -jnp.inf)

    rng_key, init_key = jax.random.split(rng_key, 2)

    # Rejection sampling
    prior_samples = cuboid_prior.sample(seed=init_key, sample_shape=(nlive,))
    mask = prior_samples['w0'] + prior_samples['wa'] < 0
    prior_samples = jax.tree.map(lambda x: x[mask], prior_samples)
    prior_samples = jax.tree.map(lambda x: x[:nlive], prior_samples)

    logl_samples = jax.vmap(logl)(prior_samples)

    # %%
    labels = [("h0rd", r"H_0r_\mathrm{d}"), (r"omegam", r"\Omega_\mathrm{m}"),
              ("w0", r"w_0"), ("wa", r"w_a")]

    return nested_sampling(
        logl, log_prior, logl_samples,
        prior_samples, nlive,
        filename, labels, rng_key)
