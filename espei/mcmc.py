"""
Legacy module for running MCMC in ESPEI

"""
import warnings
from espei.utils import database_symbols_to_fit
from espei.optimizers.opt_mcmc import EmceeOptimizer

def mcmc_fit(dbf, datasets, iterations=1000, save_interval=1, chains_per_parameter=2,
             chain_std_deviation=0.1, scheduler=None, tracefile=None, probfile=None,
             restart_trace=None, deterministic=True, prior=None, mcmc_data_weights=None):
    """
    Run MCMC via the EmceeOptimizer class

    Parameters
    ----------
    dbf : Database
        A pycalphad Database to fit with symbols to fit prefixed with `VV`
        followed by a number, e.g. `VV0001`
    datasets : PickleableTinyDB
        A database of single- and multi-phase data to fit
    iterations : int
        Number of trace iterations to calculate in MCMC. Default is 1000 iterations.
    save_interval :int
        interval of iterations to save the tracefile and probfile
    chains_per_parameter : int
        number of chains for each parameter. Must be an even integer greater or
        equal to 2. Defaults to 2.
    chain_std_deviation : float
        standard deviation of normal for parameter initialization as a fraction
        of each parameter. Must be greater than 0. Default is 0.1, which is 10%.
    scheduler : callable
        Scheduler to use with emcee. Must implement a map method.
    tracefile : str
        filename to store the trace with NumPy.save. Array has shape
        (chains, iterations, parameters)
    probfile : str
        filename to store the log probability with NumPy.save. Has shape (chains, iterations)
    restart_trace : np.ndarray
        ndarray of the previous trace. Should have shape (chains, iterations, parameters)
    deterministic : bool
        If True, the emcee sampler will be seeded to give deterministic sampling
        draws. This will ensure that the runs with the exact same database,
        chains_per_parameter, and chain_std_deviation (or restart_trace) will
        produce exactly the same results.
    prior : str
        Prior to use to generate priors. Defaults to 'zero', which keeps
        backwards compatibility. Can currently choose 'normal', 'uniform',
        'triangular', or 'zero'.
    mcmc_data_weights : dict
        Dictionary of weights for each data type, e.g. {'ZPF': 20, 'HM': 2}

    """
    warnings.warn("The mcmc convenience function will be removed in ESPEI 0.8")
    all_symbols = database_symbols_to_fit(dbf)

    optimizer = EmceeOptimizer(dbf, scheduler=scheduler)
    optimizer.save_interval = save_interval
    optimizer.fit(all_symbols, datasets, prior=prior, iterations=iterations,
                  chains_per_parameter=chains_per_parameter,
                  chain_std_deviation=chain_std_deviation,
                  deterministic=deterministic, restart_trace=restart_trace,
                  tracefile=tracefile, probfile=probfile,
                  mcmc_data_weights=mcmc_data_weights)
    optimizer.commit()
    return optimizer.dbf, optimizer.sampler

