from jax import numpy as jnp
import jax
import blackjax
from tqdm import tqdm
import anesthetic
from blackjax.ns.utils import finalise
from tensorflow_probability.substrates.jax import distributions as tfd


# Parameter registry - central definition of all priors and labels
PARAMETER_REGISTRY = {
    'h0rd': {'prior': tfd.Uniform(1_000.0, 100_000.0), 'label': r'H_0r_d'},
    'h0': {'prior': tfd.Uniform(20.0, 100.0), 'label': r'H_0'},
    'omegam': {'prior': tfd.Uniform(0.01, 0.99), 'label': r'\Omega_m'},
    'w0': {'prior': tfd.Uniform(-3.0, 1.0), 'label': r'w_0'},
    'wa': {'prior': tfd.Uniform(-3.0, 2.0), 'label': r'w_a'},
    'Mb': {'prior': tfd.Uniform(-25.0, -15.0), 'label': r'M_B'},
}


def nested_sampling(log_likelihood, log_prior, logl_samples, prior_samples,
                    nlive, labels, rng_key, **nss_kwargs):
    """Core nested sampling function - unchanged from original."""
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
    return final


def save(final, filename, labels, flatten=None):
    """Save function - unchanged from original."""
    if flatten is not None:
        particles = flatten(final.particles)
    else:
        particles = final.particles

    labels_map = {label[0]: f'${label[1]}$' for label in labels}

    samples = anesthetic.NestedSamples(
        data=particles,
        logL=final.loglikelihood,
        logL_birth=final.loglikelihood_birth,
        columns=[label[0] for label in labels],
        labels=labels_map,
    )

    print(f"anesthetic logZ = {samples.logZ():.2f} "
          f"= {samples.logL_P():.2f} - {samples.D_KL():.2f}")
    samples.to_csv(f"{filename}.csv")
    return samples, final


def sort_samples(samples):
    i = jnp.argsort(samples['a'], axis=-1, descending=True)
    samples['a'] = jnp.take_along_axis(samples['a'], i, -1)
    samples['w'] = jnp.concatenate([
        samples['w'][..., :1],
        jnp.take_along_axis(samples['w'][..., 1:-1], i, -1),
        samples['w'][..., -1:],
    ], axis=-1)
    return samples


def sampler(logl, requirements, nlive, filename, rng_key, **kwargs):
    """Build a sampler with CPL constraint: w0 + wa < 0."""

    # Build prior dictionary
    prior_dict = {}
    labels = []
    ns_kwargs = {}
    save_kwargs = {}

    for param in requirements:
        if param == 'a':
            n = kwargs['n']
            prior_dict[param] = tfd.Uniform(jnp.zeros(n-2), jnp.ones(n-2))
            labels += [(f'a{i}', f'a_{{{i}}}') for i in range(1, n-1)]
        elif param == 'w':
            n = kwargs['n']
            prior_dict[param] = tfd.Uniform(jnp.full(n, -3.0), jnp.full(n, 1.0))
            labels += [(f'w{i}', f'w_{{{i}}}') for i in range(0, n-1)]
            labels += [('wn-1', r'w_{n-1}')]
        else:
            prior_dict[param] = PARAMETER_REGISTRY[param]['prior']
            labels.append((param, PARAMETER_REGISTRY[param]['label']))

    # Create cuboid prior (before applying constraint)
    cuboid_prior = tfd.JointDistributionNamed(prior_dict)

    # Define constrained prior
    cpl = 'w0' in requirements and 'wa' in requirements
    flexknot = 'a' in requirements and 'w' in requirements

    def sample_prior(seed, sample_shape):
        keys = jax.random.split(seed, len(prior_dict))
        return dict(
            zip(prior_dict.keys(),
                [prior.sample(seed=k, sample_shape=sample_shape)
                 for k, prior in zip(keys, prior_dict.values())])
        )

    def prior_fn(x):
        return sum([
            jnp.sum(prior.log_prob(x[param])) for param, prior in prior_dict.items()
        ])

    if cpl:
        nprior = 2*nlive

        def log_prior(x):
            # Only allow w0 + wa < 0
            return jnp.where(x['w0'] + x['wa'] < 0,
                             prior_fn(x) + jnp.log(20/(20-9/2)),
                             -jnp.inf)
    else:
        nprior = nlive
        log_prior = prior_fn

    # Rejection sampling for initial points
    rng_key, init_key = jax.random.split(rng_key, 2)
    prior_samples = cuboid_prior.sample(seed=init_key, sample_shape=(nprior,))

    if cpl:
        mask = prior_samples['w0'] + prior_samples['wa'] < 0
        prior_samples = jax.tree.map(lambda x: x[mask], prior_samples)
        prior_samples = jax.tree.map(lambda x: x[:nlive], prior_samples)
    elif flexknot:
        prior_samples = sort_samples(prior_samples)

        @jax.jit
        def sorted_stepper(x, n, t):
            y = jax.tree.map(lambda x, n: x + t * n, x, n)
            y = sort_samples(y)
            return y
        ns_kwargs["stepper_fn"] = sorted_stepper

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
        save_kwargs['flatten'] = flatten

    # Evaluate initial likelihoods
    logl_samples = jax.vmap(logl)(prior_samples)

    # Run nested sampling with constrained prior
    final = nested_sampling(
        logl, log_prior, logl_samples,
        prior_samples, nlive, labels, rng_key,
        **ns_kwargs,
    )

    return save(final, filename, labels, **save_kwargs)
