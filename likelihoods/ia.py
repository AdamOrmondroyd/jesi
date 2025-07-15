import numpy as np
from jax.numpy import array, log10, einsum
from scipy.constants import c


c = c / 1000.0


class IaLogL:
    requirements = {'h0_dl_over_c'}

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

        # Compute constrained inverse C^-1_tilde using SVD for numerical stability
        # This marginalizes out nuisance parameters (H0, absolute magnitude)

        # Method 1: SVD-based pseudoinverse approach for stability
        U, s, Vt = np.linalg.svd(cov_np, full_matrices=False)

        # Regularize singular values to improve conditioning
        # Only regularize very small singular values to preserve the matrix structure
        s_min = 1e-12 * s[0]  # Relative threshold
        s_reg = np.where(s < s_min, s_min, s)

        # Compute C^-1 using regularized SVD
        S_inv = np.diag(1.0 / s_reg)
        invcov_np = Vt.T @ S_inv @ U.T

        # Compute marginalized inverse more stably
        invcov_one = invcov_np @ one_np
        one_T_invcov_one = (one_np.T @ invcov_one).item()

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
    requirements = {'h0_dl_over_c', 'h0', 'Mb'}

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
        return -0.5 * delta.T @ self.invcov @ delta + self.lognormalisation
