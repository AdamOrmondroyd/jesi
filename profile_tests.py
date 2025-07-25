import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
import matplotlib.pyplot as plt
from jax import numpy as jnp
import jax
import optax
from jayesian.nested_sampling import sampler
from jayesian.likelihoods import desidr2
from jayesian.cosmology import lcdm, cpl

rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)


def logl(x):
    # return desidr2(x, lcdm)
    return desidr2(x, cpl)


# requirements = ["h0rd", "omegam"]
# filename = "desidr2_lcdm"
requirements = ["h0rd", "omegam", "w0", "wa"]
filename = "desidr2_cpl"
nested_samples = sampler(logl, requirements, 500, filename, rng_key)

idx = jnp.argmax(nested_samples[1].loglikelihood)
print("Initial best logls:", nested_samples[1].loglikelihood[idx])
params = jax.tree.map(lambda leaf: leaf[idx], nested_samples[1].particles)


solver = optax.lbfgs()
opt_state = solver.init(params)
f = jax.jit(lambda x: -logl(x))  # We want to minimize the negative log-likelihood

val_grad = optax.value_and_grad_from_state(f)


for _ in range(50):
    value, grad = val_grad(params, state=opt_state)
    updates, opt_state = solver.update(
         grad, opt_state, params, value=value, grad=grad, value_fn=f
    )
    params = optax.apply_updates(params, updates)
    print("Current params:", params)
    print("Current value:", value)
    # print('Objective function: {:.2E}'.format(logl(params)))


a = nested_samples[0].plot_2d(list(nested_samples[1].particles.keys()), label="Posterior")
a.scatter(params, marker='*', c='C1', label="LBFGS solution", alpha=0.5)

initial_max = jax.tree.map(lambda leaf: leaf[idx], nested_samples[1].particles)
a.scatter(initial_max, marker='*', c='C2', label="NS max", alpha=0.5)
a.iloc[-1,  0].legend(loc='lower center', bbox_to_anchor=(len(a)/2, len(a)))
plt.tight_layout()
plt.savefig(f"plots/max_likelihood_{filename}.png", dpi=300, bbox_inches='tight')
