import numpy as np
from scipy.special import logsumexp
from sys import argv
from pathlib import Path
from matplotlib import pyplot as plt
import smplotlib as smpl
from anesthetic import read_chains
from anesthetic.samples import merge_samples_weighted

likelihood = argv[1]
title = argv[2] if len(argv) > 2 else likelihood
dir = Path("/Users/adam/phd/distances/chains/jovekie")
lcdm = read_chains(dir/f"{likelihood}offset/lcdm")
cpl = read_chains(dir/f"{likelihood}offset/cpl")
fks = [read_chains(dir/f"{likelihood}offset/{i}")
       for i in ['wcdm']+[f"flexknot_n={j}" for j in range(2, 21)]]
fk = merge_samples_weighted(fks)
logZfk = logsumexp([i.logZ(nsamples=1000) for i in fks], axis=0) - np.log(20)

lcdm0 = read_chains(dir/f"{likelihood}/lcdm")
cpl0 = read_chains(dir/f"{likelihood}/cpl")
fk0s = [read_chains(dir/f"{likelihood}/{i}")
        for i in ['wcdm']+[f"flexknot_n={j}" for j in range(2, 21)]]
fk0 = merge_samples_weighted(fk0s)
logZfk0 = logsumexp([i.logZ(nsamples=1000) for i in fk0s], axis=0) - np.log(20)
logbfk = logZfk - logZfk0

for m, m0 in zip([lcdm, cpl], [lcdm0, cpl0]):
    logb = m.logZ(nsamples=1000) - m0.logZ(nsamples=1000)
    print(f"{m.label} & ${m.delta_mb.mean():.3f} \\pm {m.delta_mb.std():.3f}$ & ${logb.mean():.3f} \\pm {logb.std():.3f}$ \\\\")

print(f"{fk.label} & ${fk.delta_mb.mean():.3f} \\pm {fk.delta_mb.std():.3f}$ & ${logbfk.mean():.3f} \\pm {logbfk.std():.3f}$ \\\\")
print(fk.delta_mb.mode())
