import os; os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config; config.update("jax_enable_x64", False)
import jax
from numpy import loadtxt
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from ia import IaLogL
import lcdm
from tensorflow_probability.substrates.jax import distributions as tfd
from nested_sampling import nested_sampling


# data loading stolen from Toby
path = Path('../clik_installs/desi/data/sn_data/PantheonPlus')
df = pd.read_table(path/'Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = loadtxt(path/'Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov = cov.reshape([-1, int(jnp.sqrt(len(cov)))])

logl_pantheonplus = IaLogL(df, cov, 'm_b_corr', z_cutoff=0.023)

if __name__ == "__main__":

    rng_key = jax.random.PRNGKey(1729)

    omegam_prior = tfd.Uniform(0.01, 0.99)

    prior = tfd.JointDistributionNamed(dict(omegam=omegam_prior))

    nlive = 5000
    rng_key, init_key = jax.random.split(rng_key, 2)
    prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
    logl_values = jax.vmap(lambda x: logl_pantheonplus(x, lcdm))(prior_samples)
    print(logl_values)

    nested_sampling(lambda x: logl_pantheonplus(x, lcdm), prior.log_prob,
                    prior_samples, logl_values,
                    nlive, "chains/pantheonplus",
                    [("omegam", r"\Omega_\mathrm{m}")], rng_key)
