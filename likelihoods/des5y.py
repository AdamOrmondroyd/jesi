import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
import jax
from numpy import loadtxt, argsort, sqrt, fill_diagonal
import pandas as pd
from pathlib import Path
from likelihoods.ia import IaLogL
import lcdm
from tensorflow_probability.substrates.jax import distributions as tfd
from nested_sampling import nested_sampling


# data loading stolen from Toby
path = Path(__file__).parent/'data/des5y'
df = pd.read_table(path/'DES-SN5YR_HD.csv', sep=',', engine='python')
cov = loadtxt(path/'covsys_000.txt', skiprows=1)
idx = argsort(df['zHD'])
cov = cov.reshape([-1, int(sqrt(len(cov)))])
delta = df['MUERR_FINAL'].to_numpy()
fill_diagonal(cov, delta**2 + cov.diagonal())
cov = cov[idx, :][:, idx]
df = df.iloc[idx]


logl = IaLogL(df, cov, 'MU')

if __name__ == "__main__":

    rng_key = jax.random.PRNGKey(1729)

    omegam_prior = tfd.Uniform(0.01, 0.99)

    prior = tfd.JointDistributionNamed(dict(omegam=omegam_prior))

    nlive = 5000
    rng_key, init_key = jax.random.split(rng_key, 2)
    prior_samples = prior.sample(seed=init_key, sample_shape=(nlive,))
    logl_values = jax.vmap(lambda x: logl(x, lcdm))(prior_samples)

    nested_sampling(lambda x: logl(x, lcdm), prior.log_prob,
                    prior_samples, logl_values,
                    nlive, "chains/pantheonplus",
                    [("omegam", r"\Omega_\mathrm{m}")], rng_key)
