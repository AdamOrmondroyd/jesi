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
from nested_sampling_auto import sampler
import likelihoods


rng_key = jax.random.PRNGKey(1729)
os.makedirs("chains", exist_ok=True)


def determine_requirements(model, logls):
    logl_requirements = set()
    for logl in logls:
        logl_requirements |= logl.requirements
    print(f"{logl_requirements=}")
    # if requirement is a cosmolgy function, replace requirement
    requirements = set()
    for req in logl_requirements:
        if hasattr(model, req):
            requirements |= getattr(model, req).requirements
        else:
            requirements |= {req}
    print(f"{requirements=}")
    return requirements



def main(model_name, *likelihood_names, nlive=1000, **kwargs):
    model = getattr(cosmology, model_name)
    logls = [getattr(likelihoods, name) for name in likelihood_names]
    requirements = determine_requirements(model, logls)

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
        requirements,
        nlive,
        chains_dir / f"{model_name}{kwargs_str}",
        rng_key,
        **kwargs
    )


if __name__ == "__main__":
    Fire(main)
