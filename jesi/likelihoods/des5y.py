from numpy import loadtxt, argsort, sqrt, fill_diagonal
from jax.numpy import array
import pandas as pd
from pathlib import Path
from jesi.likelihoods.ia import IaLogL, IaLogLUnmarginalised


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
