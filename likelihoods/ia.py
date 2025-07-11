from jax.numpy import array, eye, log, log10, ones, pi
from jax.numpy.linalg import solve, cholesky
from scipy.constants import c


c = c / 1000.0


def fast_constrained_quadratic_form(L_inv, y, v, v_dot_v):
    """
    Compute -y.T @ C^-1_tilde @ y using precomputed L^-1.

    Args:
        L_inv: precomputed L^-1 matrix
        y: data vector
        v: precomputed L^-1 @ one
        v_dot_v: precomputed v.T @ v

    Returns:
        Quadratic form value computed fast
    """
    # Fast matrix multiplication u = L^-1 @ y
    u = L_inv @ y

    # Compute the constrained quadratic form
    u_dot_u = (u.T @ u).squeeze()
    u_dot_v = (u.T @ v).squeeze()

    # Constrained quadratic form: -y.T @ C^-1_tilde @ y
    return -(u_dot_u - u_dot_v**2 / v_dot_v)


class IaLogL:
    def __init__(self, df, cov, mb_column, z_cutoff=0.0):

        self.df = df
        self.cov = cov

        mask = df['zHD'] > z_cutoff
        self.mb = array(df[mb_column].to_numpy()[mask])
        self.zhd = array(df['zHD'].to_numpy()[mask])
        self.zhel = array(df['zHEL'].to_numpy()[mask])

        self.cov = array(cov[mask, :][:, mask])

        # Use Cholesky decomposition for numerical stability
        self.L = cholesky(self.cov)
        self.one = ones(len(self.cov))[:, None]

        # Precompute L^-1 for fast matrix multiplication
        self.L_inv = solve(self.L, eye(len(self.cov)))
        
        # Precompute v = L^-1 @ one (constant for all likelihood evaluations)
        self.v = self.L_inv @ self.one
        self.v_dot_v = (self.v.T @ self.v).squeeze()
        
        # Compute normalization stably using Cholesky factor
        one_T_invcov_one = self.v_dot_v

        # Log normalization: -0.5 * [log|2πC| + log(1^T C^-1 1)]
        self.lognormalisation = -0.5 * (
            2 * log(self.L.diagonal()).sum()  # log|C| from Cholesky
            + log(2*pi) * len(self.cov)      # log(2π)^n
            + log(one_T_invcov_one)          # log(1^T C^-1 1)
        )

    def _y(self, params, cosmology):
        return 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)) - self.mb

    def __call__(self, params, cosmology):
        y = self._y(params, cosmology)

        # Compute constrained quadratic form fast (L_inv, v, v_dot_v precomputed)
        quadratic_form = fast_constrained_quadratic_form(self.L_inv, y, self.v, self.v_dot_v)

        result = quadratic_form / 2 + self.lognormalisation
        return result.squeeze()
