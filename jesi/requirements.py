import functools


def requires(*requirements):
    def decorator(f):
        f.requirements = set(requirements)
        return f
    return decorator


@functools.wraps(functools.partial)
def partial(f, *args, **kwargs):
    requirements = getattr(f, 'requirements', set()) | {
        r for a in args for r in getattr(a, 'requirements', set())
    }
    return requires(*requirements)(functools.partial(f, *args, **kwargs))
