from numpy import load, argsort, zeros, tril_indices, triu_indices
from numpy.linalg import inv
from jax.numpy import array
import pandas as pd
from pathlib import Path
from jesi.likelihoods.ia import IaLogL


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
