import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from anesthetic import read_chains
import smplotlib
from pypolychord.output import PolyChordOutput
from pathlib import Path

nlive = 500

logz_upperxlim = 0
kde_upperxlim = 0
fig, (axlogZ, axkde) = plt.subplots(2, figsize=(6, 9))

for chains, label, color in zip([
    "../../../resume/thesis/double/master_500",
    "../../../resume/thesis/double/no_cluster_500",
    "chains/double/double"
], [
    "PolyChord (with clustering)",
    "PolyChord (no clustering)",
    "BlackJAX Nested Slice Sampling",
], [
    "C0", "m", "c",
]):
    pclogZ = []
    pcheightratios = []
    num_in_1sigma = 0
    for i in tqdm(range(100)):
        ns = read_chains(chains + f"_{i}")
        _fig, _ax = plt.subplots()
        if "x" in ns:
            ns.x.plot.kde_1d(ax=_ax, label=label)
        else:
            ns[0].plot.kde_1d(ax=_ax, label=label)
        _ax.legend(frameon=True)
        x = _ax.lines[0].get_xdata()
        y = _ax.lines[0].get_ydata()
        yl = max(y[x < 0])
        yr = max(y[x > 0])
        pcheightratios.append(max([yl, yr]) / min([yl, yr]))

        plt.close(_fig)
        if "PolyChord" in label:
            path = Path(chains + f"_{i}")
            pc = PolyChordOutput(base_dir=str(path.parent), file_root=str(path.name))
            pclogZ.append(pc.logZ)
            if abs(pc.logZ + np.log(4)) < pc.logZerr:
                num_in_1sigma += 1
        else:
            pclogZ.append(ns.logZ())
            logZerr = ns.logZ(nsamples=100).std()
            if abs(ns.logZ() + np.log(4)) < logZerr:
                num_in_1sigma += 1

    print(f"{label}: {num_in_1sigma}/100 within 1 sigma of true logZ")
    axlogZ.hist(pclogZ, bins=20, density=True, alpha=0.5, label=label, color=color)
    axkde.hist(pcheightratios, bins=20, density=True, alpha=0.5, label=label, color=color)
axlogZ.axvline(-np.log(4), color='k', linestyle='--')
axlogZ.set(xlabel=r"$\log Z$ (true value = $-\log 4$)")
axkde.legend(fontsize='small')
axkde.set_xlim(left=1)
axkde.set(xlabel=r"$\frac{\text{larger KDE peak height}}{\text{smaller KDE peak height}}$")

fig.tight_layout()
fig.savefig("plots/double_compare.pdf", bbox_inches='tight')
plt.show()
