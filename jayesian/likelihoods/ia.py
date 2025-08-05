import numpy as np
from scipy.linalg import ldl
from jax.numpy import array, log10
from scipy.constants import c


c = c / 1000.0


class IaLogL:
    requirements = {'h0_dl_over_c'}

    def __init__(self, df, cov, mb_column, z_cutoff=0.0):

        self.df = df

        mask = df['zHD'] > z_cutoff
        self.mb = array(df[mb_column].to_numpy()[mask])
        self.zhd = array(df['zHD'].to_numpy()[mask])
        self.zhel = array(df['zHEL'].to_numpy()[mask])

        cov = cov[mask, :][:, mask]
        (self.lT, self.d, self.perm), self.lognorm = self._compute_cholesky_and_lognorm(cov)

    def _compute_cholesky_and_lognorm(self, cov):
        # Do all matrix operations in fp64 numpy for precision
        cov_np = np.array(cov, dtype=np.float64)
        one_np = np.ones((len(cov_np), 1), dtype=np.float64)

        # Compute constrained inverse C^-1_tilde for marginalization
        # This marginalizes out nuisance parameters (H0, absolute magnitude)
        invcov_np = np.linalg.inv(cov_np)
        invcov_one = np.linalg.solve(cov_np, one_np)  # More stable than inv @ one
        one_T_invcov_one = invcov_one.sum(keepdims=True)

        # Constrained inverse using Cobaya's more stable approach
        # C^-1_tilde = C^-1 - (C^-1 @ 1) @ solve(1^T @ C^-1 @ 1, (C^-1 @ 1)^T)
        invcov_tilde = (
            invcov_np
            - invcov_one @ np.linalg.solve(one_T_invcov_one, invcov_one.T)
        )

        # Compute Cholesky decomposition for GPU vmap bug fix
        # This avoids the problematic y.T @ M @ y operation
        l, d, perm = ldl(invcov_tilde)
        lT = array(l).T
        d = array(np.diag(d))

        # Compute log normalization in fp64
        sign, logdet = np.linalg.slogdet(cov_np)
        if sign != 1:
            raise ValueError("Covariance matrix must be positive definite.")
        lognorm = -0.5 * (
            logdet                           # log|C|
            + np.log(2*np.pi) * (len(cov_np) - 1)  # log(2π)^(n-1) after marginalization
            + np.log(one_T_invcov_one.item())       # log(1^T C^-1 1)
        )
        return (lT, d, perm), lognorm

    def _y(self, params, cosmology):
        return 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)
        ) - self.mb

    def __call__(self, params, cosmology):
        y = self._y(params, cosmology)

        # Use Cholesky decomposition to avoid GPU vmap bug
        # y.T @ M @ y = ||L.T @ y||² where M = L @ L.T
        v = self.lT @ y[self.perm]
        return -(v * self.d * v).sum() / 2.0 + self.lognorm


class IaLogLUnmarginalised(IaLogL):
    requirements = {'h0_dl_over_c', 'h0', 'Mb'}

    def _compute_cholesky_and_lognorm(self, cov):

        # NOTE: More stable to invert the cholesky than cholesky the inverse
        # Bear in mind that the cholesky of the inverse is the TRANSPOSE
        # of the cholesky of the inverse
        cholesky_LT = array(np.linalg.inv(
            np.linalg.cholesky(cov)
        ))

        sign, logdet = np.linalg.slogdet(cov)
        lognormalisation = -0.5 * (
            logdet
            + np.log(2*np.pi) * len(cov)
        )
        return (cholesky_LT, None, None), lognormalisation

    def _y(self, params, cosmology):
        mu = 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)
            * c / params['h0']
        ) + 25
        return self.mb - (mu + params['Mb'])

    def __call__(self, params, cosmology):
        y = self._y(params, cosmology)

        v = self.lT @ y
        return -(v * v).sum() / 2.0 + self.lognorm
