from anesthetic import read_chains

fkab = read_chains("chains/desidr2_des5y/flexknot_n=2")
fka = read_chains("chains/desidr2/flexknot_n=2")
fkb = read_chains("chains/des5y/flexknot_n=2")

cplab = read_chains("chains/desidr2_des5y/cpl")
cpla = read_chains("chains/desidr2/cpl")
cplb = read_chains("chains/des5y/cpl")

print(f"{fkab.logZ()=}")
print(f"{cplab.logZ()=}")
print(f"{fkab.logZ() - fka.logZ() - fkb.logZ()=}")
print(f"{cplab.logZ() - cpla.logZ() - cplb.logZ()=}")
print(f"{fkab.logZ(nsamples=1000).std()=}")

print(f"{fkab.logL_P()=}")
print(f"{cplab.logL_P()=}")
