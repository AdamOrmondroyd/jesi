import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config

config.update("jax_enable_x64", False)
from jax import numpy as jnp
import jax
import blackjax
from tqdm import tqdm
import anesthetic
from blackjax.ns.utils import finalise
from tensorflow_probability.substrates.jax import distributions as tfd
from nested_sampling import sample_cpl

h0rd_prior = tfd.Uniform(1_000.0, 100_000.0)  # 1/99_000
omegam_prior = tfd.Uniform(0.01, 0.99)  # 1/0.98
w0_prior = tfd.Uniform(-3.0, 1.0)  # 1/4
wa_prior = tfd.Uniform(-3.0, 2.0)  # 1/5


from desidr2 import logl_desidr2

# from pantheonplus import logl_pantheonplus
import lcdm
import cpl

rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)


# def logl(x):
#     return logl_desidr2(x, lcdm)  # + logl_pantheonplus(x, lcdm)

def logl(x):
    return logl_desidr2(x, cpl)  # + logl_pantheonplus(x, lcdm)

nested_samples = sample_cpl(logl, 500, "dp_cpl", rng_key)

idx = jnp.argmax(nested_samples[1].loglikelihood)
print("Initial best logls:", nested_samples[1].loglikelihood[idx])
params = jax.tree.map(lambda leaf: leaf[idx], nested_samples[1].particles)




# cpl_prior = tfd.JointDistributionNamed(
#     dict(
#         h0rd=h0rd_prior,
#         omegam=omegam_prior,
#         w0=w0_prior,
#         wa=wa_prior,
#     )
# )

# lcdm_prior = tfd.JointDistributionNamed(
#     dict(
#         h0rd=h0rd_prior,
#         omegam=omegam_prior,
#     )
# )

# initial_samples = lcdm_prior.sample(seed=rng_key, sample_shape=(1000,))
# initial_samples = cpl_prior.sample(seed=rng_key, sample_shape=(1000,))


# likelihood_samples = jax.vmap(logl)(initial_samples)

# idx = jnp.argmax(likelihood_samples)
# params = jax.tree.map(lambda leaf: leaf[idx], initial_samples)

import optax
solver = optax.lbfgs()
opt_state = solver.init(params)
f = jax.jit(lambda x: -logl(x))  # We want to minimize the negative log-likelihood

val_grad = optax.value_and_grad_from_state(f)


for _ in range(50):
    value, grad = val_grad(params, state =opt_state)
    updates, opt_state = solver.update(
         grad, opt_state, params, value=value, grad=grad, value_fn=f
    )
    params = optax.apply_updates(params, updates)
    print("Current params:", params)
    print("Current value:", value)
    # print('Objective function: {:.2E}'.format(logl(params)))


a = nested_samples[0].plot_2d(list(nested_samples[1].particles.keys()), label="Posterior")
a.scatter(params, marker='*', c='C1', label="LBFGS solution", alpha=0.5)

initial_max  = jax.tree.map(lambda leaf: leaf[idx], nested_samples[1].particles)
a.scatter(initial_max, marker='*', c='C2', label="NS max", alpha =0.5)
a.iloc[-1,  0].legend(loc='lower center', bbox_to_anchor=(len(a)/2, len(a)))
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig("max_likelihood.png", dpi=300, bbox_inches='tight')
