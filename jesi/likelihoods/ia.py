from jax import config
import numpy as np
from scipy.linalg import ldl
from jax.numpy import array, float32, log10, matmul
from scipy.constants import c


c = c / 1000.0


class IaLogL:
    requirements = {'h0_dl_over_c'}

    def __init__(self, df, cov, mb_column, invcov=None, z_cutoff=0.0):

        self.df = df

        mask = df['zHD'] > z_cutoff
        self.mb = array(df[mb_column].to_numpy()[mask])
        self.zhd = array(df['zHD'].to_numpy()[mask])
        self.zhel = array(df['zHEL'].to_numpy()[mask])

        cov = cov[mask, :][:, mask]
        if invcov is not None:
            invcov = invcov[mask, :][:, mask]
        (self.lT, self.d, self.perm), self.lognorm = self._compute_cholesky_and_lognorm(cov, invcov)

    def _compute_cholesky_and_lognorm(self, cov, invcov=None):
        # Do all matrix operations in fp64 numpy for precision
        cov_np = np.array(cov, dtype=np.float64)
        one_np = np.ones((len(cov_np), 1), dtype=np.float64)

        # Compute constrained inverse C^-1_tilde for marginalization
        # This marginalizes out nuisance parameters (H0, absolute magnitude)
        invcov_np = np.linalg.inv(cov_np) if invcov is None else np.array(invcov, dtype=np.float64)

        if invcov is not None:
            invcov_one = invcov_np @ one_np
            one_T_invcov_one = invcov.sum()
            invcov_tilde = invcov_np - invcov_one @ invcov_one.T / one_T_invcov_one
        else:
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
        if invcov is None:
            sign, logdet = np.linalg.slogdet(cov_np)
        else:
            sign, logdet = np.linalg.slogdet(invcov_np)
            logdet = -logdet
        if sign != 1:
            raise ValueError("Covariance matrix must be positive definite.")
        lognorm = -0.5 * (
            logdet                           # log|C|
            + np.log(2*np.pi) * (len(cov_np) - 1)  # log(2π)^(n-1) after marginalization
            + np.log(one_T_invcov_one.item())       # log(1^T C^-1 1)
        )
        if not config.jax_enable_x64:
            lognorm = float32(lognorm)
        return (lT, d, perm), lognorm

    def _y(self, params, cosmology):
        return 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)
        ) - self.mb

    def __call__(self, params, cosmology):
        y = self._y(params, cosmology)

        # Use Cholesky decomposition to avoid GPU vmap bug
        # y.T @ M @ y = ||L.T @ y||² where M = L @ L.T
        v = matmul(self.lT, y[self.perm], precision="highest")
        return -(v * self.d * v).sum() / 2.0 + self.lognorm


class IaLogLUnmarginalised(IaLogL):
    requirements = {'h0_dl_over_c', 'h0', 'Mb'}

    def __init__(self, df, cov, mb_column, invcov=None, z_cutoff=0.0):
        if mb_column == 'MU':
            raise NotImplementedError(
                "Due to sublety with the different measures on Mb and H0, "
                "Treating mu as mb is not permitted here. "
                "To recover the same result as the marginalised likelihood, "
                "use a 1/H0 prior."
            )
        return super().__init__(df, cov, mb_column, invcov, z_cutoff)

    def _compute_cholesky_and_lognorm(self, cov, invcov=None):

        # NOTE: More stable to invert the cholesky than cholesky the inverse
        # Bear in mind that the cholesky of the inverse is the TRANSPOSE
        # of the inverse of the cholesky
        if invcov is not None:
            cholesky_LT = np.linalg.cholesky(invcov).T
            sign, logdet = np.linalg.slogdet(invcov)
            logdet = -logdet
        else:
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
