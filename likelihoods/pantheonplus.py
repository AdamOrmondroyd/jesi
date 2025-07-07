import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
from numpy import loadtxt
import jax.numpy as jnp
import pandas as pd
from pathlib import Path
from likelihoods.ia import IaLogL


# data loading stolen from Toby
path = Path(__file__).parent/'data/pantheonplus'
df = pd.read_table(path/'Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = loadtxt(path/'Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov = cov.reshape([-1, int(jnp.sqrt(len(cov)))])

logl = IaLogL(df, cov, 'm_b_corr', z_cutoff=0.023)
