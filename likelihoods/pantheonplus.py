from numpy import loadtxt, sqrt
import pandas as pd
from pathlib import Path
from likelihoods.ia import IaLogL, IaLogLUnmarginalised


# data loading stolen from Toby
path = Path(__file__).parent/'data/pantheonplus'
df = pd.read_table(path/'Pantheon+SH0ES.dat', sep=' ', engine='python')
cov = loadtxt(path/'Pantheon+SH0ES_STAT+SYS.cov', skiprows=1)
cov = cov.reshape([-1, int(sqrt(len(cov)))])

logl = IaLogL(df, cov, 'm_b_corr', z_cutoff=0.023)

loglunmarginalised = IaLogLUnmarginalised(df, cov, 'm_b_corr', z_cutoff=0.023)
