# first line: 59
        @wraps(f)
        def inner_f(*args, **kwargs):
            extra_args = len(args) - len(all_args)
            if extra_args <= 0:
                return f(*args, **kwargs)

            # extra_args > 0
            args_msg = ['{}={}'.format(name, arg)
                        for name, arg in zip(kwonly_args[:extra_args],
                                             args[-extra_args:])]
            args_msg = ", ".join(args_msg)
            warnings.warn(f"Pass {args_msg} as keyword args. From version "
                          f"{version} passing these as positional arguments "
                          "will result in an error", FutureWarning)
            kwargs.update(zip(sig.parameters, args))
            return f(**kwargs)
