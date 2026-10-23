from jesi.likelihoods import desidr2, des5y
from jesi.cosmology import cpl, flexknot
from jax.numpy import array

params = {
    "omegam": 0.3,
    "h0rd": 10000.0,
    "w0": -1.0,
    "wa": 0.5,
    "w": array([-1.0, -0.5]),
    "a": array([]),
}

print(f"{desidr2(params, flexknot)=}")
print(f"{desidr2(params, cpl)=}")
print(f"{des5y(params, flexknot)=}")
print(f"{des5y(params, cpl)=}")
