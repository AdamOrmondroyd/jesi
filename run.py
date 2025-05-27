# %%
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
from blackjax.ns.utils import finalise
tfd = tfp.distributions

rng_key = jax.random.PRNGKey(0)

h0rd_prior = tfd.Uniform(1000, 100_000)  # 
omegam_prior = tfd.Uniform(0.01, 0.99)  # 1/0.98
w0_prior = tfd.Uniform(-3.0, 1.0)  # 1/4
wa_prior = tfd.Uniform(-3.0, 2.0)

sort_of_prior = tfd.JointDistributionNamed(dict(
    h0rd=h0rd_prior,
    omegam=omegam_prior,
    w0=w0_prior,
    wa=wa_prior,
))


logv = jnp.log((20-9/2) * (0.99-0.01) * (100_000 - 100))


def log_prob(x):
    # return -logv if within the prior bounds, else -inf
    return jnp.where(x['w0'] + x['wa'] < 1,
                     sort_of_prior.log_prob(x) + jnp.log(20/(20-9/2)),
                     -jnp.inf)
    # w0 = x['w0']
    # wa = x['wa']
    # omegam = x['omegam']
    # h0rd = x['h0rd']
#
    # return jnp.where(
    #     w0 + wa < 1 and -3 <= w0, # <= 1,  # and -3 <= wa <= 2 and
    #     #0.01 <= omegam <= 0.99 and 1000 <= h0rd <= 100_000,
    #     -logv,
    #     -jnp.inf,
    # )

# NOTE: do not use this for the prior probabilities
prior_sample_generator = tfd.JointDistributionNamed(dict(
    h0rd=h0rd_prior,
    omegam=omegam_prior,
    w0=w0_prior,
    wa=lambda w0: tfd.Uniform(-3.0, jnp.minimum(2.0, 1-w0))
))


test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior_sample_generator.sample(seed=jax.random.PRNGKey(0))
)

# %%
nlive = 5000
nprior = 10 * nlive
n_delete = nlive // 2
rng_key, init_key = jax.random.split(rng_key, 2)

ns = blackjax.ns.adaptive.nss(
    logprior_fn=log_prob,
    loglikelihood_fn=logl_desidr2,
    n_delete=n_delete,
    num_mcmc_steps=4*3,
    ravel_fn=ravel_fn,
)

dead = []
# %%


def integrate(ns, rng_key):
    rng_key, init_key = jax.random.split(rng_key, 2)
    particles = prior_sample_generator.sample(seed=init_key, sample_shape=(nprior,))
    print(particles)
    print(f"{particles['w0'].shape=}")
    assert jnp.all(particles['w0'] + particles['wa'] < 1)
    prior_values = jax.vmap(log_prob)(particles)
    print(f"{prior_values=}")

    logl = jax.vmap(logl_desidr2)(particles)
    from scipy.special import logsumexp
    print(f"{logl=}")
    print(f"logZ = {logsumexp(logl)-jnp.log(nprior)}")
    top_logl, idx = jax.lax.top_k(logl, nlive)
    state = ns.init(jax.tree.map(lambda x: x[idx], particles), top_logl)

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

    return state, finalise(state, dead), particles, logl


# %%
state, final, cold_particles, cold_logl = integrate(ns, rng_key)

cold_theta = np.vstack(
    [cold_particles['h0rd'],
     cold_particles['omegam'],
     cold_particles['w0'],
     cold_particles['wa'],
     ])
theta = np.vstack(
    [final.particles['h0rd'],
     final.particles['omegam'],
     final.particles['w0'],
     final.particles['wa'],
     ])

theta = np.concatenate([cold_theta, theta], axis=-1)
logl = np.concatenate([cold_logl, final.logL])
# cold points have logl_birth -ing
logl_birth = np.concatenate([jnp.full_like(cold_logl, -jnp.inf), final.logL_birth])
print(theta.shape)
print(logl.shape)
print(logl_birth.shape)

labels = [("h0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}"),
          ("w0", r"w_0"), ("wa", r"w_a")]
print(labels)
labels_map = {l[0]: '$'+l[1]+'$' for l in labels}

samples = anesthetic.NestedSamples(
    data=theta.T,
    logL=logl,
    logL_birth=logl_birth,
    columns=[l[0] for l in labels],
    labels=labels_map,
)
samples.to_csv("chains/cpl.csv")
print(samples)

print(f"log Z = {state.sampler_state.logZ}")

# %%
