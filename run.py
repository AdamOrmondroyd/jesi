import os
if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    from jax import config

    config.update("jax_enable_x64", False)
from pathlib import Path
from fire import Fire
from functools import partial
import jax
import cosmology
import nested_sampling
import likelihoods


rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)


def main(model_name, *likelihood_names, nlive=1000, **kwargs):
    model = getattr(cosmology, model_name)
    sampler = getattr(nested_sampling, f"sample_{model_name}")
    logls = [getattr(likelihoods, name) for name in likelihood_names]

    def logl(x, model, logls):
        return sum(logl(x, model) for logl in logls)

    logl = partial(logl, model=model, logls=logls)

    root = "_".join(likelihood_names)
    kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())
    if kwargs_str:
        kwargs_str = "_" + kwargs_str
    chains_dir = Path("chains") / root
    chains_dir.mkdir(exist_ok=True, parents=True)

    sampler(
        logl,
        nlive,
        chains_dir / f"{model_name}{kwargs_str}",
        rng_key,
        **kwargs
    )


if __name__ == "__main__":
    Fire(main)
