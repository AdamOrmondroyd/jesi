from matplotlib import pyplot as plt
from anesthetic import read_chains
import smplotlib

cpl = read_chains("chains/desidr2_des5y/cpl")
jfk = read_chains("chains/desidr2_des5y/flexknot_n=2")
fk = read_chains("chains/csd3/desidr2_des5y_wide_2")
fk['wn-1'] = fk.wn

fk['wa'] = fk['wn-1'] - fk.w0
jfk['wa'] = jfk['wn-1'] - jfk.w0
cpl['wn-1'] = cpl.w0 + cpl.wa

alpha_scatter = 0.5
alpha_cplkde = 0.8
alpha_fkkde = 0.6
cplcolor = 'C1'
fkcolor = '#58acbc'

fig, ax = plt.subplots(2, 1, figsize=(5, 9))

cpl.prior().plot.scatter_2d('w0', 'wa', ax=ax[0], color=cplcolor, label='CPL prior', alpha=alpha_scatter, ncompress=1000)
jfk.prior().plot.scatter_2d('w0', 'wa', ax=ax[0], color=fkcolor, label='FlexKnot n=2 prior', alpha=alpha_scatter, ncompress=1000)
cpl.plot.kde_2d('w0', 'wa', ax=ax[0], color=cplcolor, alpha=alpha_cplkde, label='CPL posterior')
fk.plot.kde_2d('w0', 'wa', ax=ax[0], color=fkcolor, alpha=alpha_fkkde, label='FlexKnot n=2 posterior')
ax[0].set_xlabel('$w_0$')
ax[0].set_ylabel('$w_a$')
ax[0].legend(fontsize='small')
ax[0].scatter([-1], [0], color='k', marker='+', lw=1.5)
ax[0].set(xlim=(-3, 1), ylim=(-4, 4))

cpl.prior().plot.scatter_2d('w0', 'wn-1', ax=ax[1], color=cplcolor, label='CPL prior', alpha=alpha_scatter, ncompress=1000)
jfk.prior().plot.scatter_2d('w0', 'wn-1', ax=ax[1], color=fkcolor, label='FlexKnot n=2 prior', alpha=alpha_scatter, ncompress=1000)
cpl.plot.kde_2d('w0', 'wn-1', ax=ax[1], color=cplcolor, alpha=alpha_cplkde)
fk.plot.kde_2d('w0', 'wn-1', ax=ax[1], color=fkcolor, alpha=alpha_fkkde)
ax[1].scatter([-1], [-1], color='k', marker='+', lw=1.5)
ax[1].set_xlabel('$w_0$')
ax[1].set_ylabel('$w_{n-1}$')
ax[1].set(xlim=(-3, 1), ylim=(-6, 1))
fig.suptitle("Comparison of CPL and FlexKnot n=2 priors")
fig.tight_layout()
fig.savefig("plots/cplprior.pdf", bbox_inches='tight')
plt.show()
