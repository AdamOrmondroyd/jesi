import os
if "Darwin" == os.uname().sysname:
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
from jax import config
config.update("jax_enable_x64", False)
from pathlib import Path  # noqa: E402
from fire import Fire  # noqa: E402
from functools import partial  # noqa: E402
import jax  # noqa: E402
from jesi import cosmology  # noqa: E402
from jesi import likelihoods  # noqa: E402
from jesi.nested_sampling import sampler  # noqa: E402

print(f"JAX in {'64' if config.jax_enable_x64 else '32'}-bit mode")
print(f"JAX platform: {os.environ.get('JAX_PLATFORM_NAME', 'default')}")

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
