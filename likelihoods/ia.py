from jax.numpy import array, log, log10, ones, pi, eye
from jax.numpy.linalg import inv, slogdet, solve
from scipy.constants import c


c = c / 1000.0


class IaLogL:
    def __init__(self, df, cov, mb_column, z_cutoff=0.0,
                 h0min=20, h0max=100):

        self.df = df
        self.cov = cov

        mask = df['zHD'] > z_cutoff
        self.mb = array(df[mb_column].to_numpy()[mask])
        self.zhd = array(df['zHD'].to_numpy()[mask])
        self.zhel = array(df['zHEL'].to_numpy()[mask])

        self.cov = array(cov[mask, :][:, mask])

        one = ones(len(self.cov))[:, None]

        # Use solve instead of inv for numerical stability
        invcov_one = solve(self.cov, one)
        one_T_invcov_one = (one.T @ invcov_one).squeeze()

        # Still need full inverse for invcov_tilde computation
        invcov = solve(self.cov, eye(len(self.cov)))

        self.invcov_tilde = (
            invcov - (invcov_one @ invcov_one.T) / one_T_invcov_one
        )

        self.lognormalisation = 0.5 * (
            log(2*pi) - slogdet(2 * pi * self.cov)[1]
            - log(one_T_invcov_one)
        ) + log(c / (1e-5 * (h0max - h0min)))

        self.h0min = h0min
        self.h0max = h0max
        self.log_scale_factor = log(1e-5 / c)
        self.onesigma_times_5_over_log10 = (
            one.T @ self.invcov_tilde * 5 / log(10)
        )

    def _y(self, params, cosmology):
        return 5 * log10(
            cosmology.h0_dl_over_c(self.zhd, self.zhel, params)) - self.mb

    def __call__(self, params, cosmology):
        y = self._y(params, cosmology)
        capital_y = self.onesigma_times_5_over_log10 @ y

        n = capital_y + 1

        log_term = (
            n * (self.log_scale_factor + log(self.h0max))
            + log(1 - (self.h0min/self.h0max)**n) - log(n)
        )
        result = (
            - y.T @ self.invcov_tilde @ y / 2
            + log_term
            + self.lognormalisation)
        return result.squeeze()
