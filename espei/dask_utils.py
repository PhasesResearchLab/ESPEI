"""
Parallelization with dask, which is an optional dependency and is only imported when a dask scheduler is used.
"""

try:
    import dask
    from distributed import Client
except ImportError as e:
    raise ImportError("Parallelizing ESPEI with dask requires the dask and distributed packages. "
                      "Install them or set `mcmc.scheduler: null` to run on a single core.") from e
from emcee.ensemble import _function_wrapper

# Force distributed's work-stealing to be False
dask.config.set({'distributed.scheduler.work-stealing': False})


class ImmediateClient(Client):
    """
    A subclass of distributed.Client that automatically unwraps the Futures
    returned by map.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The active memory manager (AMM) removes any duplicate data found across workers,
        # and any tasks that require said data will go to the one worker where it is retained on
        # However, we want the context to remain on all workers to limit data transfer so we disable AMM
        _client = super(ImmediateClient, self)
        _client.amm.stop()
        self.future_kwargs = {}

    def map(self, f, *iterators, **kwargs):
        """Map a function on a sequence of arguments.

        Any keyword arguments are passed to distributed.Client.map
        """
        _client = super(ImmediateClient, self)

        # This is specific to emcee, where f, args, kwargs are put into a function wrapper object
        # We want to submit the kwargs to the client before evaluating func which allows us
        # to reuse the submitted context data
        # NOTE: in emcee 3.x, _function_wrapper has been renamed to FunctionWrapper
        if isinstance(f, _function_wrapper):
            func = f.f
            # Submit kwargs as futures if not done so
            # If a future in the context was cancelled, then also resubmit it
            #   This is in case dask workers are restarted, they could still have access to the context
            for key in f.kwargs:
                if key not in self.future_kwargs or self.future_kwargs[key].cancelled():
                    self.future_kwargs[key] = _client.submit(lambda x: x, f.kwargs[key], key=key)
            result = _client.gather(_client.map(func, *[list(it) for it in iterators], **self.future_kwargs))
        else:
            result = _client.gather(_client.map(f, *[list(it) for it in iterators], **kwargs))
        return result
