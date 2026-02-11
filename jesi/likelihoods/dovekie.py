from numpy import load, argsort, zeros, tril_indices, triu_indices
from numpy.linalg import inv
import pandas as pd
from pathlib import Path
from jesi.likelihoods.ia import IaLogL
from jesi.likelihoods.des5y import GeorgeIaLogL


# data loading stolen from Toby
path = Path(__file__).parent/'data/dovekie'
df = pd.read_table(path/'DES-Dovekie_HD.csv', sep='\\s+', engine='python', skiprows=8)

d = load(path/"STAT+SYS.npz")
nsn = d['nsn'].item()
invcov = zeros((nsn, nsn))
invcov[triu_indices(nsn)] = d['cov']

# Reflect to lower triangular part to make it symmetric
i_lower = tril_indices(nsn, -1)
invcov[i_lower] = invcov.T[i_lower]

cov = inv(invcov)

idx = argsort(df['zHD'])
invcov = invcov[idx, :][:, idx]
cov = cov[idx, :][:, idx]
df = df.iloc[idx]

logl = IaLogL(df, cov, 'MU', invcov)

# george fiddle
des_id = 10.0
george_mask = df['IDSURVEY'] != des_id

logloffset = GeorgeIaLogL(george_mask, df, cov, mb_column='MU', z_cutoff=0.0)
