import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import jax
from tensorflow_probability.substrates.jax import distributions as tfd
import blackjax
from blackjax.ns.utils import finalise
from jax.scipy.stats import norm
from tqdm import tqdm


rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)

x_prior = tfd.Uniform(-1.0, 1.0)
# y_prior = tfd.Uniform(-1.0, 1.0)

prior = tfd.JointDistributionNamed(dict(
    x=x_prior,
    # y=y_prior,
))


def logl(x): return norm.logpdf(**x)
# def logl(x): return norm.logpdf((x['x']**2 + x['y']**2)**0.5)


nlive = 5000

rng_key, init_key = jax.random.split(rng_key, 2)

prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
logl_samples = jax.vmap(logl)(prior_samples)


def nested_sampling(log_likelihood, log_prior, nlive, rng_key,
                    prior_samples, logl_samples,
                    ):
    n_delete = nlive // 2
    dead = []

    ns = blackjax.nss(
        logprior_fn=log_prior,
        loglikelihood_fn=log_likelihood,
        num_delete=n_delete,
        num_inner_steps=8,
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


nested_sampling(
    logl, prior.log_prob, nlive, rng_key,
    prior_samples, logl_samples)
