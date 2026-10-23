"""
Create a Hubble residual diagram comparing DES-5Y and DES-Dovekie
supernova data relative to Planck flat ΛCDM.
"""
# HACK:
# Need to move this to a version with dovekie to actually plot, popping it in
# this worktree for safekeeping

import matplotlib.pyplot as plt
import smplotlib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.integrate import quad
from scipy.stats import binned_statistic


def load_des5y_data():
    """Load DES-5Y supernova data."""
    path = Path('jesi/likelihoods/data/des5y/DES-SN5YR_HD.csv')
    df = pd.read_csv(path)
    return df

def load_dovekie_data():
    """Load DES-Dovekie supernova data."""
    path = Path('jesi/likelihoods/data/dovekie/DES-Dovekie_HD.csv')
    df = pd.read_table(path, sep=r'\s+', engine='python', skiprows=8)
    return df


def mu_theory(z, Om=0.315):
    """Distance modulus for flat ΛCDM (up to an additive constant).

    The absolute calibration (M_B / H0) is degenerate, so we fit for
    an offset when computing residuals.
    """
    def E_inv(z):
        return 1.0 / np.sqrt(Om * (1 + z)**3 + (1 - Om))

    dc, _ = quad(E_inv, 0, z, limit=100)
    dl = dc * (1 + z)
    return 5 * np.log10(dl)


mu_theory_vec = np.vectorize(mu_theory)


def residuals(z, mu):
    """Compute mu - mu_theory, absorbing the absolute offset via least squares."""
    mu_th = mu_theory_vec(z)
    offset = np.mean(mu - mu_th)
    return mu - mu_th - offset


def binned_residuals(z, res, bins):
    """Compute binned mean residuals and standard error on the mean."""
    mean, bin_edges, _ = binned_statistic(z, res, statistic='mean', bins=bins)
    std, _, _ = binned_statistic(z, res, statistic='std', bins=bins)
    count, _, _ = binned_statistic(z, res, statistic='count', bins=bins)
    sem = std / np.sqrt(count)
    bin_centres = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric mean
    return bin_centres, mean, sem


def create_hubble_diagram():
    """Create Hubble residual diagram for DES-5Y vs DES-Dovekie."""

    des5y_df = load_des5y_data()
    dovekie_df = load_dovekie_data()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Residuals
    des5y_z = des5y_df['zHD'].values
    des5y_res = residuals(des5y_z, des5y_df['MU'].values)

    dovekie_z = dovekie_df['zHD'].values
    dovekie_res = residuals(dovekie_z, dovekie_df['MU'].values)

    # Log-spaced bins
    z_all = np.concatenate([des5y_z, dovekie_z])
    bins = np.logspace(np.log10(z_all.min()), np.log10(z_all.max()), 20)

    # Binned means with error on the mean
    d5_zc, d5_mean, d5_sem = binned_residuals(des5y_z, des5y_res, bins)
    dv_zc, dv_mean, dv_sem = binned_residuals(dovekie_z, dovekie_res, bins)

    ax.errorbar(d5_zc, d5_mean, yerr=d5_sem, fmt='s', ms=4, capsize=2,
                color='C3', label=f'DES-5Y ({len(des5y_z)} SNe)')
    ax.errorbar(dv_zc, dv_mean, yerr=dv_sem, fmt='d', ms=4, capsize=2,
                color='C0', label=f'DES-Dovekie ({len(dovekie_z)} SNe)')

    ax.axhline(0, color='k', linewidth=0.5)

    ax.set_xlabel('Redshift $z$')
    ax.set_ylabel(r'$\mu - \mu_{\Lambda\mathrm{CDM}}(\Omega_\mathrm{m}=0.315)$')
    ax.legend(loc='upper right')
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig('hubble_diagram.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    create_hubble_diagram()
