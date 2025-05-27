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
import lcdm
from blackjax.ns.utils import finalise
from jax.scipy.special import logsumexp
tfd = tfp.distributions

rng_key = jax.random.PRNGKey(1729)

h0rd_prior = tfd.Uniform(1000, 100_000)
omegam_prior = tfd.Uniform(0.01, 0.99)

prior = tfd.JointDistributionNamed(dict(
    h0rd=h0rd_prior,
    omegam=omegam_prior,
))

test_sample, ravel_fn = jax.flatten_util.ravel_pytree(
    prior.sample(seed=jax.random.PRNGKey(0))
)

# %%
nlive = 5000
n_delete = nlive // 2
rng_key, init_key = jax.random.split(rng_key, 2)

ns = blackjax.ns.adaptive.nss(
    logprior_fn=prior.log_prob,
    loglikelihood_fn=lambda x: logl_desidr2(x, lcdm),
    n_delete=n_delete,
    num_mcmc_steps=4*3,
    ravel_fn=ravel_fn,
)

dead = []
# %%

nmonaco = 100_000
monaco = logsumexp(
    jax.vmap(lambda x: logl_desidr2(x, lcdm))(prior.sample(seed=rng_key, sample_shape=(nmonaco,)))
) - jnp.log(nmonaco)
print(f"logZ = {monaco}")


def integrate(ns, rng_key):
    rng_key, init_key = jax.random.split(rng_key, 2)
    particles = prior.sample(seed=init_key, sample_shape=(nlive,))
    print(particles)
    prior_values = jax.vmap(prior.log_prob)(particles)
    print(f"{prior_values=}")

    logl = jax.vmap(lambda x: logl_desidr2(x, lcdm))(particles)
    print(f"{logl=}")

    state = ns.init(particles, logl)

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


# %%
state, final = integrate(ns, rng_key)

theta = np.vstack(
    [final.particles['h0rd'],
     final.particles['omegam'],
     ])

logl = final.logL
# cold points have logl_birth -inf
logl_birth = final.logL_birth
print(theta.shape)
print(logl.shape)
print(logl_birth.shape)

labels = [("h0rd", r"H_0r_\mathrm{d}"), (r"Omegam", r"\Omega_\mathrm{m}")]
print(labels)
labels_map = {l[0]: f'${l[1]}$' for l in labels}

samples = anesthetic.NestedSamples(
    data=theta.T,
    logL=logl,
    logL_birth=logl_birth,
    columns=[l[0] for l in labels],
    labels=labels_map,
)
samples.to_csv("chains/lcdm.csv")
print(samples)

print(f"sampler logZ = {state.sampler_state.logZ:.2f}")

print(f"anesthetic logZ = {samples.logZ():.2f} = {samples.logL_P():.2f} - {samples.D_KL():.2f}")

print(f"Monte Carlo logZ = {monaco:.2f}")
