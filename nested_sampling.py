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


def nested_sampling(log_likelihood, log_prior, logl_samples, prior_samples,
                    nlive, filename, labels, rng_key, flatten=None, **nss_kwargs,
                    ):
    n_delete = nlive // 2
    dead = []

    ns = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=3*len(labels),
        **nss_kwargs,
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

    if flatten is not None:
        particles = flatten(final.particles)
    else:
        particles = final.particles

    labels_map = {l[0]: f'${l[1]}$' for l in labels}

    samples = anesthetic.NestedSamples(
        data=particles,
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


def sample_flexknot(logl, nlive, filename, rng_key, n):

    a_prior = tfd.Uniform(jnp.zeros(n-2), jnp.ones(n-2))
    w_prior = tfd.Uniform(-3*jnp.ones(n), jnp.zeros(n))

    def sample_prior(seed, sample_shape):
        k1, k2, k3, k4 = jax.random.split(seed, 4)
        return {
            'h0rd': h0rd_prior.sample(sample_shape, seed=k1),
            'omegam': omegam_prior.sample(sample_shape, seed=k2),
            'a': a_prior.sample(sample_shape, seed=k3),
            'w': w_prior.sample(sample_shape, seed=k4),
        }

    def prior_fn(x):
        prior = jnp.sum(h0rd_prior.log_prob(x['h0rd']))
        prior += jnp.sum(omegam_prior.log_prob(x['omegam']))
        prior += jnp.sum(a_prior.log_prob(x['a']))
        prior += jnp.sum(w_prior.log_prob(x['w']))
        return prior

    rng_key, init_key = jax.random.split(rng_key, 2)

    def sort_samples(samples):
        i = jnp.argsort(samples['a'], axis=-1, descending=True)
        samples['a'] = jnp.take_along_axis(samples['a'], i, -1)
        samples['w'] = jnp.concatenate([
            samples['w'][..., :1],
            jnp.take_along_axis(samples['w'][..., 1:-1], i, -1),
            samples['w'][..., -1:],
        ], axis=-1)
        return samples
    prior_samples = sample_prior(seed=init_key, sample_shape=(2*nlive,))
    prior_samples = sort_samples(prior_samples)
    logl_samples = jax.vmap(logl)(prior_samples)

    @jax.jit
    def wrapped_stepper(x, n, t):
        y = jax.tree.map(lambda x, n: x + t * n, x, n)
        y = sort_samples(y)
        return y

    labels = [
        ("h0rd", r"H_0r_\mathrm{d}"),
         (r"omegam", r"\Omega_\mathrm{m}")
    ] + [
        (f"a{i}", f"a_{{{i}}}") for i in range(1, n-1)
    ] + [
        (f"w{i}", f"w_{{{i}}}") for i in range(n)
    ]

    def flatten(particles):
        data_dict = {}
        for key, values in particles.items():
            if key in ['a', 'w']:  # vector parameters
                if key == 'a':
                    # a1, a2, a3, ... (skip a0 since it's fixed at 1)
                    for i in range(values.shape[1]):
                        data_dict[f'{key}{i+1}'] = values[:, i]
                else:  # w
                    # w0, w1, w2, w3, ... wn
                    for i in range(values.shape[1]):
                        data_dict[f'{key}{i}'] = values[:, i]
            else:
                # scalar parameters
                data_dict[key] = values
        return data_dict

    return nested_sampling(
        logl, prior_fn, logl_samples,
        prior_samples, nlive,
        filename, labels, rng_key,
        flatten=flatten,
        stepper_fn=wrapped_stepper,
    )
