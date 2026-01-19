from numpy import load, loadtxt, argsort, sqrt, fill_diagonal, zeros, tril_indices, triu_indices
from numpy.linalg import inv
from jax.numpy import array
import pandas as pd
from pathlib import Path
from jesi.likelihoods.ia import IaLogL, IaLogLUnmarginalised


# data loading stolen from Toby
path = Path(__file__).parent/'data/dovekie'
df = pd.read_table(path/'DES-Dovekie_HD.csv', sep='\\s+', engine='python', skiprows=8)

d = load(path/"STAT+SYS.npz")
nsn = d['nsn'].item()
inv_cov = zeros((nsn, nsn))
inv_cov[triu_indices(nsn)] = d['cov']

# Reflect to lower triangular part to make it symmetric
i_lower = tril_indices(nsn, -1)
inv_cov[i_lower] = inv_cov.T[i_lower]

idx = argsort(df['zHD'])
invcov = inv_cov[idx, :][:, idx]
df = df.iloc[idx]
cov = inv(inv_cov)

logl = IaLogL(df, cov, 'MU')

loglunmarginalised = IaLogLUnmarginalised(df, cov, 'MU')

# george fiddle
des_id = 10.0
george_mask = df['IDSURVEY'] != des_id


class GeorgeIaLogL(IaLogL):
    requirements = {'h0_dl_over_c', 'delta_mb'}

    def __init__(self, george_mask, *args, **kwargs):
        self.george_mask = array(george_mask.to_numpy())
        super().__init__(*args, **kwargs)

    def _y(self, params, cosmology):
        # offset is the george offset
        y = super()._y(params, cosmology)
        y -= params['delta_mb'] * self.george_mask
        return y


logloffset = GeorgeIaLogL(george_mask, df, cov, mb_column='MU', z_cutoff=0.0)
