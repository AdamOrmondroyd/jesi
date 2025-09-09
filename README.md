# JESI
### Dark Energy Jaxoscopic Instrument
BAO and SNe Ia likelihoods for `JAX`.

![jesi logo](logo.png)

Remember on cuda to install the appropriate jax:
```bash
pip install "jax[cuda12]==0.5.2"
pip install -e .
# or
pip install -e ".[cuda12]"
```

### Nested sampling
Call `run.py` followed by the model and datasets. Optionally specify `nlive` (default is 1000) and any additional parameters for the model.
For example, flexknot requires the number of knots `n`:
```bash
python run.py lcdm des5y
python run.py cpl desidr2 pantheonplus --nlive=500
python run.py flexknot desidr2 --n=10
```
