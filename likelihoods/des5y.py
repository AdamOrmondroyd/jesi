import os

if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
from numpy import loadtxt, argsort, sqrt, fill_diagonal
import pandas as pd
from pathlib import Path
from likelihoods.ia import IaLogL


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
