import numpy as np
from jax.numpy import array, log10, einsum
from scipy.constants import c


c = c / 1000.0


class IaLogL:
    def __init__(self, df, cov, mb_column, z_cutoff=0.0):

        self.df = df
        self.cov = cov

        mask = df['zHD'] > z_cutoff
        self.mb = array(df[mb_column].to_numpy()[mask])
        self.zhd = array(df['zHD'].to_numpy()[mask])
        self.zhel = array(df['zHEL'].to_numpy()[mask])

        # Do all matrix operations in fp64 numpy for precision
        cov_np = np.array(cov[mask, :][:, mask], dtype=np.float64)
        one_np = np.ones((len(cov_np), 1), dtype=np.float64)

        # Compute constrained inverse C^-1_tilde in fp64
        # This marginalizes out nuisance parameters (H0, absolute magnitude)
        invcov_np = np.linalg.inv(cov_np)
        invcov_one = np.linalg.solve(cov_np, one_np)  # More stable than inv @ one
        one_T_invcov_one = invcov_one.sum()

        # Constrained inverse: C^-1_tilde = C^-1 - (C^-1 @ 1)(1^T @ C^-1) / (1^T @ C^-1 @ 1)
        self.invcov_tilde = array(
            invcov_np - (invcov_one @ invcov_one.T) / one_T_invcov_one)

        # Compute log normalization in fp64
        sign, logdet = np.linalg.slogdet(cov_np)
        self.lognormalisation = -0.5 * (
            logdet                           # log|C|
            + np.log(2*np.pi) * (len(cov_np) - 1)  # log(2π)^(n-1) after marginalization
            + np.log(one_T_invcov_one)       # log(1^T C^-1 1)
        )

    def _y(self, params, cosmology):
        return 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)
        ) - self.mb

    def __call__(self, params, cosmology):
        y = self._y(params, cosmology)

        # Fast quadratic form with precisely computed constrained inverse
        quadratic_form = -einsum('i,ij,j', y.squeeze(), self.invcov_tilde, y.squeeze())

        result = quadratic_form / 2 + self.lognormalisation
        return result


class IaLogLUnmarginalised:
    def __init__(self, df, cov, mb_column, z_cutoff=0.0):

        self.df = df
        self.cov = cov

        mask = df['zHD'] > z_cutoff
        self.mb = array(df[mb_column].to_numpy()[mask])
        self.zhd = array(df['zHD'].to_numpy()[mask])
        self.zhel = array(df['zHEL'].to_numpy()[mask])

        cov_np = np.array(cov[mask, :][:, mask])

        self.invcov = array(np.linalg.inv(cov_np))

        # Compute log normalization in fp64
        sign, logdet = np.linalg.slogdet(cov_np)
        self.lognormalisation = -0.5 * (
            logdet
            + np.log(2*np.pi) * len(cov_np)
        )

    def delta(self, params, cosmology):
        mu = 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)
            * c / params['h0']
        ) + 25
        return self.mb - (mu + params['Mb'])

    def __call__(self, params, cosmology):
        delta = self.delta(params, cosmology)

        # add log10 for now to account for Mb prior
        return -0.5 * delta.T @ self.invcov @ delta + self.lognormalisation + log(10)
