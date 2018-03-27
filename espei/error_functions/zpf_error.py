"""
Calculate error due to ZPF tielines.

The general approach is similar to the PanOptimizer rough search method.

1. With all phases active, calculate the chemical potentials of the tieline
   endpoints via ``equilibrium`` calls. Done in ``estimate_hyperplane``.
2. Calculate the target chemical potentials, which are the average chemical
   potentials of all of the current chemical potentials at the tieline endpoints.
3. Calculate the current chemical potentials of the desired single phases
4. The error is the difference between these chemical potentials

There's some special handling for tieline endpoints where we do not know the
composition conditions to calculate chemical potentials at.
"""

import textwrap, time, operator, logging
from collections import defaultdict, OrderedDict

import numpy as np
import dask
import tinydb

from pycalphad import calculate, equilibrium, variables as v

def estimate_hyperplane(dbf, comps, phases, current_statevars, comp_dicts, phase_models, parameters,
                        massfuncs=None, massgradfuncs=None,
                        callables=None, grad_callables=None,
                        hess_callables=None,
                        ):
    """
    Calculate the chemical potentials for a hyperplane, one vertex at a time

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    current_statevars : dict
        Dictionary of state variables, e.g. v.P and v.T, no compositions.
    comp_dicts : list
        List of tuples of composition dictionaries and phase flags. Composition
        dictionaries are pycalphad variable dicts and the flag is a string e.g.
        ({v.X('CU'): 0.5}, 'disordered')
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    massfuncs : dict
        Callables of mass derivatives to pass to pycalphad
    massgradfuncs : dict
        Gradient callables of mass derivatives to pass to pycalphad
    callables : dict
        Callables to pass to pycalphad
    grad_callables : dict
        Gradient callables to pass to pycalphad
    hess_callables : dict
        Hessian callables to pass to pycalphad

    Returns
    -------
    numpy.ndarray
        Array of chemical potentials.

    Notes
    -----
    This takes just *one* set of phase equilibria, e.g. a dataset point of
    [['FCC_A1', ['CU'], [0.1]], ['LAVES_C15', ['CU'], [0.3]]]
    and calculates the chemical potentials given all the phases possible at the
    given compositions. Then the average chemical potentials of each end point
    are taken as the target hyperplane for the given equilibria.

    """
    region_chemical_potentials = []
    parameters = OrderedDict(sorted(parameters.items(), key=str))
    # TODO: unclear whether we use phase_flag and how it would be used. It should be just a 'disordered' kind of flag.
    for cond_dict, phase_flag in comp_dicts:
        # We are now considering a particular tie vertex
        for key, val in cond_dict.items():
            if val is None:
                cond_dict[key] = np.nan
        cond_dict.update(current_statevars)
        if np.any(np.isnan(list(cond_dict.values()))):
            # This composition is unknown -- it doesn't contribute to hyperplane estimation
            pass
        else:
            # Extract chemical potential hyperplane from multi-phase calculation
            # Note that we consider all phases in the system, not just ones in this tie region
            multi_eqdata = equilibrium(dbf, comps, phases, cond_dict, verbose=False,
                                       model=phase_models, scheduler=dask.local.get_sync, parameters=parameters,
                                       massfuncs=massfuncs,
                                       massgradfuncs=massgradfuncs,
                                       callables=callables,
                                       grad_callables=grad_callables,
                                       hess_callables=hess_callables,
                                       )
            # Does there exist only a single phase in the result with zero internal degrees of freedom?
            # We should exclude those chemical potentials from the average because they are meaningless.
            num_phases = len(np.squeeze(multi_eqdata['Phase'].values != ''))
            zero_dof = np.all((multi_eqdata['Y'].values == 1.) | np.isnan(multi_eqdata['Y'].values))
            if (num_phases == 1) and zero_dof:
                region_chemical_potentials.append(np.full_like(np.squeeze(multi_eqdata['MU'].values), np.nan))
            else:
                region_chemical_potentials.append(np.squeeze(multi_eqdata['MU'].values))
    region_chemical_potentials = np.nanmean(region_chemical_potentials, axis=0, dtype=np.float)
    return region_chemical_potentials


def tieline_error(dbf, comps, current_phase, cond_dict, region_chemical_potentials, phase_flag,
                  phase_models, parameters, debug_mode=False,
                  massfuncs=None, massgradfuncs=None,
                  callables=None, grad_callables=None, hess_callables=None,
                  ):
    """

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    current_phase : list
        List of phases to consider
    current_statevars : dict
        Dictionary of state variables, e.g. v.P and v.T, no compositions.
    comp_dicts : list
        List of tuples of composition dictionaries and phase flags. Composition
        dictionaries are pycalphad variable dicts and the flag is a string e.g.
        ({v.X('CU'): 0.5}, 'disordered')
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    massfuncs : dict
        Callables of mass derivatives to pass to pycalphad
    massgradfuncs : dict
        Gradient callables of mass derivatives to pass to pycalphad
    callables : dict
        Callables to pass to pycalphad
    grad_callables : dict
        Gradient callables to pass to pycalphad
    hess_callables : dict
        Hessian callables to pass to pycalphad
    cond_dict :
    region_chemical_potentials : numpy.ndarray
        Array of chemical potentials for target equilibrium hyperplane.
    phase_flag : str
        String of phase flag, e.g. 'disordered'.
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    debug_mode : bool
        If True, will write out scripts when pycalphad fails to find a stable
        equilibrium. These scripts can be used to debug pycalphad.
    massfuncs : dict
        Callables of mass derivatives to pass to pycalphad
    massgradfuncs : dict
        Gradient callables of mass derivatives to pass to pycalphad
    callables : dict
        Callables to pass to pycalphad
    grad_callables : dict
        Gradient callables to pass to pycalphad
    hess_callables : dict
        Hessian callables to pass to pycalphad

    Returns
    -------
    float
        Single value for the total error between the current hyperplane and target hyperplane.

    """
    if np.any(np.isnan(list(cond_dict.values()))):
        # We don't actually know the phase composition here, so we estimate it
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P],
                                  model=phase_models, parameters=parameters, pdens=100,
                                  massfuncs=massfuncs, callables=callables,
                                  )
        driving_force = np.multiply(region_chemical_potentials,
                                    single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(driving_force.max())
    elif phase_flag == 'disordered':
        # Construct disordered sublattice configuration from composition dict
        # Compute energy
        # Compute residual driving force
        # TODO: Check that it actually makes sense to declare this phase 'disordered'
        num_dof = sum([len(set(c).intersection(comps)) for c in dbf.phases[current_phase].constituents])
        desired_sitefracs = np.ones(num_dof, dtype=np.float)
        dof_idx = 0
        for c in dbf.phases[current_phase].constituents:
            dof = sorted(set(c).intersection(comps))
            if (len(dof) == 1) and (dof[0] == 'VA'):
                return 0
            # If it's disordered config of BCC_B2 with VA, disordered config is tiny vacancy count
            sitefracs_to_add = np.array([cond_dict.get(v.X(d)) for d in dof],
                                        dtype=np.float)
            # Fix composition of dependent component
            sitefracs_to_add[np.isnan(sitefracs_to_add)] = 1 - np.nansum(sitefracs_to_add)
            desired_sitefracs[dof_idx:dof_idx + len(dof)] = sitefracs_to_add
            dof_idx += len(dof)
        single_eqdata = calculate(dbf, comps, [current_phase],
                                  T=cond_dict[v.T], P=cond_dict[v.P], points=desired_sitefracs,
                                  model=phase_models, parameters=parameters, massfuncs=massfuncs,
                                  callables=callables,)
        driving_force = np.multiply(region_chemical_potentials, single_eqdata['X'].values).sum(axis=-1) - single_eqdata['GM'].values
        error = float(np.squeeze(driving_force))
    else:
        # Extract energies from single-phase calculations
        single_eqdata = equilibrium(dbf, comps, [current_phase], cond_dict, verbose=False,
                                    model=phase_models,
                                    scheduler=dask.local.get_sync, parameters=parameters,
                                    massfuncs=massfuncs,
                                    massgradfuncs=massgradfuncs,
                                    callables=callables,
                                    grad_callables=grad_callables,
                                    hess_callables=hess_callables,
                                    )
        if np.all(np.isnan(single_eqdata['NP'].values)):
            error_time = time.time()
            template_error = """
            from pycalphad import Database, equilibrium
            from pycalphad.variables import T, P, X
            import dask
            dbf_string = \"\"\"
            {0}
            \"\"\"
            dbf = Database(dbf_string)
            comps = {1}
            phases = {2}
            cond_dict = {3}
            parameters = {4}
            equilibrium(dbf, comps, phases, cond_dict, scheduler=dask.local.get_sync, parameters=parameters)
            """
            template_error = textwrap.dedent(template_error)
            if debug_mode:
                logging.warning('Dumping', 'error-'+str(error_time)+'.py')
                with open('error-'+str(error_time)+'.py', 'w') as f:
                    f.write(template_error.format(dbf.to_string(fmt='tdb'), comps, [current_phase], cond_dict, {key: float(x) for key, x in parameters.items()}))
        # Sometimes we can get a miscibility gap in our "single-phase" calculation
        # Choose the weighted mixture of site fractions
            logging.debug('Calculation failure: all NaN phases with conditions: {}'.format(cond_dict))
            return 0
        select_energy = float(single_eqdata['GM'].values)
        region_comps = []
        for comp in [c for c in sorted(comps) if c != 'VA']:
            region_comps.append(cond_dict.get(v.X(comp), np.nan))
        region_comps[region_comps.index(np.nan)] = 1 - np.nansum(region_comps)
        error = np.multiply(region_chemical_potentials, region_comps).sum() - select_energy
        error = float(error)
    return error


def calculate_zpf_error(dbf, comps, phases, datasets, phase_models, parameters=None, scheduler=None,
                    massfuncs=None, massgradfuncs=None,
                    callables=None, grad_callables=None, hess_callables=None,
                    ):
    """
    Calculate error due to phase equilibria data

    Parameters
    ----------
    dbf : pycalphad.Database
        Database to consider
    comps : list
        List of active component names
    phases : list
        List of phases to consider
    datasets : espei.utils.PickleableTinyDB
        Datasets that contain single phase data
    phase_models : dict
        Phase models to pass to pycalphad calculations
    parameters : dict
        Dictionary of symbols that will be overridden in pycalphad.equilibrium
    scheduler : class
        Scheduler implementing a get_sync method
    massfuncs : dict
        Callables of mass derivatives to pass to pycalphad
    massgradfuncs : dict
        Gradient callables of mass derivatives to pass to pycalphad
    callables : dict
        Callables to pass to pycalphad
    grad_callables : dict
        Gradient callables to pass to pycalphad
    hess_callables : dict
        Hessian callables to pass to pycalphad

    Returns
    -------
    list
        List of errors from phase equilibria data

    """
    desired_data = datasets.search((tinydb.where('output') == 'ZPF') &
                                   (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

    def safe_index(itms, idxx):
        try:
            return itms[idxx]
        except IndexError:
            return None

    errors = []
    for data in desired_data:
        payload = data['values']
        conditions = data['conditions']
        data_comps = list(set(data['components']).union({'VA'}))
        # create a dictionary of each set of phases containing a list of individual points on the tieline
        # individual tieline points are tuples of (conditions, {composition dictionaries})
        phase_regions = defaultdict(lambda: list())
        # TODO: Fix to only include equilibria listed in 'phases'
        for idx, p in enumerate(payload):
            phase_key = tuple(sorted(rp[0] for rp in p))
            if len(phase_key) < 2:
                # Skip single-phase regions for fitting purposes
                continue
            # Need to sort 'p' here so we have the sorted ordering used in 'phase_key'
            # rp[3] optionally contains additional flags, e.g., "disordered", to help the solver
            comp_dicts = [(dict(zip([v.X(x.upper()) for x in rp[1]], rp[2])), safe_index(rp, 3))
                          for rp in sorted(p, key=operator.itemgetter(0))]
            cur_conds = {}
            for key, value in conditions.items():
                value = np.atleast_1d(np.asarray(value))
                if len(value) > 1:
                    value = value[idx]
                cur_conds[getattr(v, key)] = float(value)
            phase_regions[phase_key].append((cur_conds, comp_dicts))
        # for each set of phases in equilibrium and their individual tieline points
        for region, region_eq in phase_regions.items():
            # for each tieline region conditions and compositions
            for current_statevars, comp_dicts in region_eq:
                # a "region" is a set of phase equilibria
                region_chemical_potentials = estimate_hyperplane(dbf, data_comps, phases, current_statevars, comp_dicts,
                                                      phase_models, parameters, massfuncs=massfuncs, massgradfuncs=massgradfuncs,
                                                      callables=callables, grad_callables=grad_callables, hess_callables=hess_callables,)
                # Now perform the equilibrium calculation for the isolated phases and add the result to the error record
                for current_phase, cond_dict in zip(region, comp_dicts):
                    # TODO: Messy unpacking
                    cond_dict, phase_flag = cond_dict
                    # We are now considering a particular tie vertex
                    for key, val in cond_dict.items():
                        if val is None:
                            cond_dict[key] = np.nan
                    cond_dict.update(current_statevars)
                    errors.append(tieline_error(dbf, data_comps, current_phase, cond_dict, region_chemical_potentials, phase_flag,
                                                        phase_models, parameters, massfuncs=massfuncs, massgradfuncs=massgradfuncs,
                                                        callables=callables, grad_callables=grad_callables, hess_callables=hess_callables,))
    return errors

