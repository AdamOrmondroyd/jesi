from jax.numpy import array, log, log10, ones, pi
from jax.numpy.linalg import inv, slogdet
from scipy.constants import c


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
        invcov = inv(self.cov)
        self.invcov_tilde = (
            invcov - invcov @ one @ one.T @ invcov / (one.T @ invcov @ one)
        )
        self.lognormalisation = 0.5 * (
            log(2*pi) - slogdet(2 * pi * self.cov)[1]
            - log((one.T @ invcov @ one).squeeze())
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

        log_term = (capital_y + 1) * self.log_scale_factor + log(
            self.h0max**(capital_y + 1)-self.h0min**(capital_y + 1)
        ) - log(capital_y + 1)
        result = (
            - y.T @ self.invcov_tilde @ y / 2
            + log_term
            + self.lognormalisation)
        return result.squeeze()
