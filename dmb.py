from sys import argv
from matplotlib import pyplot as plt
import smplotlib as smpl
from anesthetic import read_chains

likelihood = argv[1]
title = argv[2] if len(argv) > 2 else likelihood

# lcdm = read_chains(f"chains/{likelihood}offset/lcdm")
# cpl = read_chains(f"chains/{likelihood}offset/cpl")
lcdm = read_chains(f"/Users/adam/phd/distances/chains/jovekie/{likelihood}/lcdm")
cpl = read_chains(f"/Users/adam/phd/distances/chains/jovekie/{likelihood}/cpl")

fig, ax = plt.subplots(figsize=(4, 4))

lcdm.delta_mb.plot.hist_1d(ax=ax, label=r"$\Lambda$CDM", color="C0", alpha=0.6, density=True)
cpl.delta_mb.plot.hist_1d(ax=ax, label="CPL", color="#76cd26", alpha=0.6, density=True)
ax.axvline(-0.04, color="k", ls="--")
ax.set_xlabel(r"$\Delta m_\mathrm{B}$")
ax.set_xlim(-0.1, 0.1)
ax.legend()
ax.set_title(title)
fig.savefig(f"plots/dmb_{likelihood}.pdf", bbox_inches="tight")
plt.show()

# lcdm0 = read_chains(f"chains/{likelihood}/lcdm")
# cpl0 = read_chains(f"chains/{likelihood}/cpl")
# for m, m0 in zip([lcdm, cpl], [lcdm0, cpl0]):
#     logb = m.logZ(nsamples=1000) - m0.logZ(nsamples=1000)
#     print(f"{m.label} & ${m.delta_mb.mean():.3f} \\pm {m.delta_mb.std():.3f}$ & ${logb.mean():.3f} \\pm {logb.std():.3f}$ \\\\")
