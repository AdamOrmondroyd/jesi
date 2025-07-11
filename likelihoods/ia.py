from jax.numpy import array, log, log10, ones, pi
from jax.numpy.linalg import solve, cholesky
from scipy.constants import c


c = c / 1000.0


def stable_constrained_quadratic_form(L, y, one):
    """
    Compute -y.T @ C^-1_tilde @ y stably using Cholesky decomposition.

    Args:
        L: Cholesky factor of covariance matrix C (lower triangular)
        y: data vector
        one: vector of ones

    Returns:
        Quadratic form value computed stably
    """
    # Solve L @ u = y and L @ v = one
    u = solve(L, y)
    v = solve(L, one)

    # Compute the constrained quadratic form
    # This is equivalent to -y.T @ C^-1_tilde @ y where C^-1_tilde
    # is the constrained inverse with the marginalization over nuisance parameters
    u_dot_u = (u.T @ u).squeeze()
    u_dot_v = (u.T @ v).squeeze()
    v_dot_v = (v.T @ v).squeeze()

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

        # Compute normalization stably using Cholesky factor
        v = solve(self.L, self.one)
        one_T_invcov_one = (v.T @ v).squeeze()

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

        # Compute constrained quadratic form stably
        quadratic_form = stable_constrained_quadratic_form(self.L, y, self.one)

        result = quadratic_form / 2 + self.lognormalisation
        return result.squeeze()
