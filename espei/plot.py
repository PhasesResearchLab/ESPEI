"""
Plotting of input data and calculated database quantities
"""
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import tinydb
from symengine import Symbol
from pycalphad import Model, calculate, equilibrium, variables as v
from pycalphad.core.utils import unpack_species
from pycalphad.plot.utils import phase_legend
from pycalphad.plot.eqplot import eqplot, _map_coord_to_variable, unpack_condition

from espei.error_functions.non_equilibrium_thermochemical_error import get_prop_samples, get_sample_condition_dicts
from espei.utils import bib_marker_map
from espei.core_utils import get_prop_data, filter_configurations, filter_temperatures, symmetry_filter, ravel_zpf_values
from espei.sublattice_tools import canonicalize, recursive_tuplify, endmembers_from_interaction
from espei.utils import build_sitefractions

plot_mapping = {
    'T': 'Temperature (K)',
    'P': 'Pressure (Pa)',
    'CPM': 'Heat Capacity (J/K-mol-atom)',
    'HM': 'Enthalpy (J/mol-atom)',
    'SM': 'Entropy (J/K-mol-atom)',
    'CPM_FORM': 'Heat Capacity of Formation (J/K-mol-atom)',
    'HM_FORM': 'Enthalpy of Formation (J/mol-atom)',
    'SM_FORM': 'Entropy of Formation (J/K-mol-atom)',
    'CPM_MIX': 'Heat Capacity of Mixing (J/K-mol-atom)',
    'HM_MIX': 'Enthalpy of Mixing (J/mol-atom)',
    'SM_MIX': 'Entropy of Mixing (J/K-mol-atom)'
}


def dataplot(comps, phases, conds, datasets, tielines=True, ax=None, legend_generator = phase_legend, plot_kwargs=None, tieline_plot_kwargs=None) -> plt.Axes:
    """
    Plot datapoints corresponding to the components, phases, and conditions.


    Parameters
    ----------
    comps : list
        Names of components to consider in the calculation.
    phases : []
        Names of phases to consider in the calculation.
    conds : dict
        Maps StateVariables to values and/or iterables of values.
    datasets : PickleableTinyDB
    tielines : bool
        If True (default), plot the tie-lines from the data
    ax : matplotlib.Axes
        Default axes used if not specified.
    legend_generator: callable
        Function that creates legend handles and colors for list of phases.
    plot_kwargs : dict
        Additional keyword arguments to pass to the matplotlib plot function for points
    tieline_plot_kwargs : dict
        Additional keyword arguments to pass to the matplotlib plot function for tielines

    Returns
    -------
    matplotlib.Axes
        A plot of phase equilibria points as a figure

    Examples
    --------

    >>> from espei.datasets import load_datasets, recursive_glob  # doctest: +SKIP
    >>> from espei.plot import dataplot  # doctest: +SKIP
    >>> datasets = load_datasets(recursive_glob('.', '*.json'))  # doctest: +SKIP
    >>> my_phases = ['BCC_A2', 'CUMG2', 'FCC_A1', 'LAVES_C15', 'LIQUID']  # doctest: +SKIP
    >>> my_components = ['CU', 'MG' 'VA']  # doctest: +SKIP
    >>> conditions = {v.P: 101325, v.T: (500, 1000, 10), v.X('MG'): (0, 1, 0.01)}  # doctest: +SKIP
    >>> dataplot(my_components, my_phases, conditions, datasets)  # doctest: +SKIP

    """
    indep_comps = [key for key, value in conds.items() if isinstance(key, v.X) and len(np.atleast_1d(value)) > 1]
    indep_pots = [key for key, value in conds.items() if ((key == v.T) or (key == v.P)) and len(np.atleast_1d(value)) > 1]
    plot_kwargs = plot_kwargs or {}
    phases = sorted(phases)

    # determine what the type of plot will be
    if len(indep_comps) == 1 and len(indep_pots) == 1:
        projection = None
    elif len(indep_comps) == 2 and len(indep_pots) == 0:
        projection = 'triangular'
    else:
        raise ValueError('The eqplot projection is not defined and cannot be autodetected. There are {} independent compositions and {} indepedent potentials.'.format(len(indep_comps), len(indep_pots)))

    if projection is None:
        x = indep_comps[0].species.name
        y = indep_pots[0]
    elif projection == 'triangular':
        x = indep_comps[0].species.name
        y = indep_comps[1].species.name

    # set up plot if not done already
    if ax is None:
        ax = plt.subplot(projection=projection)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True)
        plot_title = '-'.join([component.title() for component in sorted(comps) if component != 'VA'])
        ax.set_title(plot_title, fontsize=20)
        ax.set_xlabel('X({})'.format(x), labelpad=15, fontsize=20)
        ax.set_xlim((0, 1))
        if projection is None:
            ax.set_ylabel(plot_mapping.get(str(y), y), fontsize=20)
        elif projection == 'triangular':
            ax.set_ylabel('X({})'.format(y), labelpad=15, fontsize=20)
            ax.set_ylim((0, 1))
            ax.yaxis.label.set_rotation(60)
            # Here we adjust the x coordinate of the ylabel.
            # We make it reasonably comparable to the position of the xlabel from the xaxis
            # As the figure size gets very large, the label approaches ~0.55 on the yaxis
            # 0.55*cos(60 deg)=0.275, so that is the xcoord we are approaching.
            ax.yaxis.label.set_va('baseline')
            fig_x_size = ax.figure.get_size_inches()[0]
            y_label_offset = 1 / fig_x_size
            ax.yaxis.set_label_coords(x=(0.275 - y_label_offset), y=0.5)

    output = 'ZPF'
    # TODO: used to include VA. Should this be added by default. Can't determine presence of VA in eq.
    # Techincally, VA should not be present in any phase equilibria.
    # For now, don't get datasets that are a subset of the current system because this breaks mass balance assumptions in ravel_zpf_values
    desired_data = datasets.search((tinydb.where('output') == output) &
                                   (tinydb.where('components').test(lambda x: (set(x).issubset(comps + ['VA'])) and (len(set(x) - {'VA'}) == (len(indep_comps) + 1)))) &
                                   (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))

    # get all the possible references from the data and create the bibliography map
    bib_reference_keys = sorted({entry.get('reference', '') for entry in desired_data})
    symbol_map = bib_marker_map(bib_reference_keys)

    # The above handled the phases as in the equilibrium, but there may be
    # phases that are in the datasets but not in the equilibrium diagram that
    # we would like to plot point for (they need color maps).
    # To keep consistent colors with the equilibrium diagram, we will append
    # the new phases from the datasets to the existing phases in the equilibrium
    # calculation.
    data_phases = set()
    for entry in desired_data:
        data_phases.update(set(entry['phases']))
    new_phases = sorted(list(data_phases.difference(set(phases))))
    phases.extend(new_phases)
    # matplotlib doesn't allow legend labels to have leading underscores,
    # so generate the legend without them, special casing only hyperplane for now
    # https://github.com/matplotlib/matplotlib/issues/5200/
    phases_without_hyperplane = [phase_name for phase_name in phases if phase_name != "__HYPERPLANE__"]
    legend_handles, phase_color_map = phase_legend(phases_without_hyperplane)
    # Force hyperplane color to black
    phase_color_map["__HYPERPLANE__"] = "black"

    if projection is None:
        # TODO: There are lot of ways this could break in multi-component situations
        # plot x (X) vs. y (T/P)
        y = str(y)
        # handle plotting kwargs
        scatter_kwargs = {'markersize': 6, 'markeredgewidth': 1}
        # raise warnings if any of the aliased versions of the default values are used
        possible_aliases = [('markersize', 'ms'), ('markeredgewidth', 'mew')]
        for actual_arg, aliased_arg in possible_aliases:
            if aliased_arg in plot_kwargs:
                warnings.warn("'{0}' passed as plotting keyword argument to dataplot, but the alias '{1}' is already set to '{2}'. Use the full version of the keyword argument '{1}' to override the default.".format(aliased_arg, actual_arg, scatter_kwargs.get(actual_arg)))
        scatter_kwargs.update(plot_kwargs)

        # fixed_pot_conds used to filter the ZPF values for ones that match the
        # caller specified conditions (via floating point equality [==])
        fixed_pot_conds = {str(key): val for key, val in conds.items() if ((key == v.T) or (key == v.P)) and len(np.atleast_1d(val)) == 1}
        eq_dict = ravel_zpf_values(desired_data, [x], conditions=fixed_pot_conds)
        updated_tieline_plot_kwargs = {'linewidth':1, 'color':'k'}
        if tieline_plot_kwargs is not None:
            updated_tieline_plot_kwargs.update(tieline_plot_kwargs)
        # No special case handling for number of phases in equilibrium.
        # It is up to dataset curators to provide data that thermodynamically correct to plot.
        equilibria_to_plot = [phase_region for phase_regions in eq_dict.values() for phase_region in phase_regions]
        for eq in equilibria_to_plot:
            # plot the scatter points for the right phases
            x_points, y_points = [], []
            for phase_name, comp_dict, ref_key in eq:
                sym_ref = symbol_map[ref_key]
                x_val, y_val = comp_dict[x], comp_dict[y]
                if x_val is not None and y_val is not None:
                    ax.plot(x_val, y_val,
                            label=sym_ref['formatted'],
                            fillstyle=sym_ref['markers']['fillstyle'],
                            marker=sym_ref['markers']['marker'],
                            linestyle='',
                            color=phase_color_map[phase_name],
                            **scatter_kwargs)
                x_points.append(x_val)
                y_points.append(y_val)

            if tielines and len(x_points) > 1:
                # plot the tielines
                if all([xx is not None and yy is not None for xx, yy in zip(x_points, y_points)]):
                    ax.plot(x_points, y_points, **updated_tieline_plot_kwargs)

    elif projection == 'triangular':
        scatter_kwargs = {'markersize': 4, 'markeredgewidth': 0.4}
        # raise warnings if any of the aliased versions of the default values are used
        possible_aliases = [('markersize', 'ms'), ('markeredgewidth', 'mew')]
        for actual_arg, aliased_arg in possible_aliases:
            if aliased_arg in plot_kwargs:
                warnings.warn("'{0}' passed as plotting keyword argument to dataplot, but the alias '{1}' is already set to '{2}'. Use the full version of the keyword argument '{1}' to override the default.".format(aliased_arg, actual_arg, scatter_kwargs.get(actual_arg)))
        scatter_kwargs.update(plot_kwargs)

        # Axes must both be molar quantities, so assume that P and T potentials are not fixed
        eq_dict = ravel_zpf_values(desired_data, [x, y], {'T': conds[v.T], 'P': conds[v.P]})

        # two phase
        updated_tieline_plot_kwargs = {'linewidth':1, 'color':'k'}
        if tieline_plot_kwargs is not None:
            updated_tieline_plot_kwargs.update(tieline_plot_kwargs)
        equilibria_to_plot = eq_dict.get(1, [])
        equilibria_to_plot.extend(eq_dict.get(2, []))
        for eq in equilibria_to_plot: # list of things in equilibrium
            # plot the scatter points for the right phases
            x_points, y_points = [], []
            for phase_name, comp_dict, ref_key in eq:
                sym_ref = symbol_map[ref_key]
                x_val, y_val = comp_dict[x], comp_dict[y]
                if x_val is not None and y_val is not None:
                    ax.plot(x_val, y_val,
                            label=sym_ref['formatted'],
                            fillstyle=sym_ref['markers']['fillstyle'],
                            marker=sym_ref['markers']['marker'],
                            linestyle='',
                            color=phase_color_map[phase_name],
                            **scatter_kwargs)
                x_points.append(x_val)
                y_points.append(y_val)

            if tielines and len(x_points) > 1:
                # plot the tielines
                if all([xx is not None and yy is not None for xx, yy in zip(x_points, y_points)]):
                    ax.plot(x_points, y_points, **updated_tieline_plot_kwargs)


        # three phase
        updated_tieline_plot_kwargs = {'linewidth':1, 'color':'r'}
        if tieline_plot_kwargs is not None:
            updated_tieline_plot_kwargs.update(tieline_plot_kwargs)
        for eq in eq_dict.get(3,[]): # list of things in equilibrium
            # plot the scatter points for the right phases
            x_points, y_points = [], []
            for phase_name, comp_dict, ref_key in eq:
                if phase_name == "__HYPERPLANE__":
                    # Don't plot the overall compositions for tie-triangles
                    continue
                x_val, y_val = comp_dict[x], comp_dict[y]
                x_points.append(x_val)
                y_points.append(y_val)
            # Make sure the triangle completes
            x_points.append(x_points[0])
            y_points.append(y_points[0])
            # plot
            # check for None values
            if all([xx is not None and yy is not None for xx, yy in zip(x_points, y_points)]):
                ax.plot(x_points, y_points, **updated_tieline_plot_kwargs)

    # now we will add the symbols for the references to the legend handles
    for ref_key in bib_reference_keys:
        mark = symbol_map[ref_key]['markers']
        # The legend marker edge width appears smaller than in the plot.
        # We will add this small hack to increase the width in the legend only.
        legend_kwargs = scatter_kwargs.copy()
        legend_kwargs['markeredgewidth'] = 1
        legend_kwargs['markersize'] = 6
        legend_handles.append(mlines.Line2D([], [], linestyle='',
                                            color='black', markeredgecolor='black',
                                            label=symbol_map[ref_key]['formatted'],
                                            fillstyle=mark['fillstyle'],
                                            marker=mark['marker'],
                                            **legend_kwargs))

    # finally, add the completed legend
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))

    return ax


def _get_interaction_predicted_values(dbf, comps, phase_name, configuration, output):
    mod = Model(dbf, comps, phase_name)
    mod.models['idmix'] = 0  # TODO: better reference state handling
    endpoints = endmembers_from_interaction(configuration)
    first_endpoint = _translate_endmember_to_array(endpoints[0], mod.ast.atoms(v.SiteFraction))
    second_endpoint = _translate_endmember_to_array(endpoints[1], mod.ast.atoms(v.SiteFraction))
    grid = np.linspace(0, 1, num=100)
    point_matrix = grid[None].T * second_endpoint + (1 - grid)[None].T * first_endpoint
    # TODO: Real temperature support
    predicted_values = calculate(
        dbf, comps, [phase_name], output=output,
        T=298.15, P=101325, points=point_matrix, model=mod)[output].values.flatten()
    return grid, predicted_values


def plot_interaction(dbf, comps, phase_name, configuration, output, datasets=None, symmetry=None, ax=None, plot_kwargs=None, dataplot_kwargs=None) -> plt.Axes:
    """
    Return one set of plotted Axes with data compared to calculated parameters

    Parameters
    ----------
    dbf : Database
        pycalphad thermodynamic database containing the relevant parameters.
    comps : Sequence[str]
        Names of components to consider in the calculation.
    phase_name : str
        Name of the considered phase phase
    configuration : Tuple[Tuple[str]]
        ESPEI-style configuration
    output : str
        Model property to plot on the y-axis e.g. ``'HM_MIX'``, or ``'SM_MIX'``.
        Must be a ``'_MIX'`` property.
    datasets : tinydb.TinyDB
    symmetry : list
        List of lists containing indices of symmetric sublattices e.g. [[0, 1], [2, 3]]
    ax : plt.Axes
        Default axes used if not specified.
    plot_kwargs : Optional[Dict[str, Any]]
        Keyword arguments to ``ax.plot`` for the predicted data.
    dataplot_kwargs : Optional[Dict[str, Any]]
        Keyword arguments to ``ax.plot`` the observed data.

    Returns
    -------
    plt.Axes

    """
    if not output.endswith('_MIX'):
        raise ValueError("`plot_interaction` only supports HM_MIX, SM_MIX, or CPM_MIX outputs.")
    if not plot_kwargs:
        plot_kwargs = {}
    if not dataplot_kwargs:
        dataplot_kwargs = {}

    if not ax:
        ax = plt.subplot()

    # Plot predicted values from the database
    grid, predicted_values = _get_interaction_predicted_values(dbf, comps, phase_name, configuration, output)
    plot_kwargs.setdefault('label', 'This work')
    plot_kwargs.setdefault('color', 'k')
    ax.plot(grid, predicted_values, **plot_kwargs)

    # Plot the observed values from the datasets
    # TODO: model exclusions handling
    # TODO: better reference state handling
    mod_srf = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
    mod_srf.models = {'ref': mod_srf.models['ref']}

    # _MIX assumption
    prop = output.split('_MIX')[0]
    desired_props = (f"{prop}_MIX", f"{prop}_FORM")
    if datasets is not None:
        solver_qry = (tinydb.where('solver').test(symmetry_filter, configuration, recursive_tuplify(symmetry) if symmetry else symmetry))
        desired_data = get_prop_data(comps, phase_name, desired_props, datasets, additional_query=solver_qry)
        desired_data = filter_configurations(desired_data, configuration, symmetry)
        desired_data = filter_temperatures(desired_data)
    else:
        desired_data = []

    species = unpack_species(dbf, comps)
    # phase constituents are Species objects, so we need to be doing intersections with those
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active
    constituents = [[sp.name for sp in sorted(subl_constituents.intersection(species))] for subl_constituents in phase_constituents]
    subl_dof = list(map(len, constituents))
    calculate_dict = get_prop_samples(desired_data, constituents)
    sample_condition_dicts = get_sample_condition_dicts(calculate_dict, canonicalize(constituents, symmetry), phase_name)
    interacting_subls = [c for c in recursive_tuplify(configuration) if isinstance(c, tuple)]
    if (len(set(interacting_subls)) == 1) and (len(interacting_subls[0]) == 2):
        # This configuration describes all sublattices with the same two elements interacting
        # In general this is a high-dimensional space; just plot the diagonal to see the disordered mixing
        endpoints = endmembers_from_interaction(configuration)
        endpoints = [endpoints[0], endpoints[-1]]
        disordered_config = True
    else:
        disordered_config = False
    bib_reference_keys = sorted({entry.get('reference', '') for entry in desired_data})
    symbol_map = bib_marker_map(bib_reference_keys)
    for data in desired_data:
        indep_var_data = None
        response_data = np.zeros_like(data['values'], dtype=np.float64)
        if disordered_config:
            # Take the second element of the first interacting sublattice as the coordinate
            # Because it's disordered all sublattices should be equivalent
            # TODO: Fix this to filter because we need to guarantee the plot points are disordered
            occ = data['solver']['sublattice_occupancies']
            subl_idx = np.nonzero([isinstance(c, (list, tuple)) for c in occ[0]])[0]
            if len(subl_idx) > 1:
                subl_idx = int(subl_idx[0])
            else:
                subl_idx = int(subl_idx)
            indep_var_data = [c[subl_idx][1] for c in occ]
        else:
            interactions = np.array([cond_dict[Symbol('YS')] for cond_dict in sample_condition_dicts])
            indep_var_data = 1 - (interactions+1)/2
        if data['output'].endswith('_FORM'):
            # All the _FORM data we have still has the lattice stability contribution
            # Need to zero it out to shift formation data to mixing
            temps = data['conditions'].get('T', 298.15)
            pressures = data['conditions'].get('P', 101325)
            points = build_sitefractions(phase_name, data['solver']['sublattice_configurations'],
                                            data['solver']['sublattice_occupancies'])
            for point_idx in range(len(points)):
                missing_variables = mod_srf.ast.atoms(v.SiteFraction) - set(points[point_idx].keys())
                # Set unoccupied values to zero
                points[point_idx].update({key: 0 for key in missing_variables})
                # Change entry to a sorted array of site fractions
                points[point_idx] = list(OrderedDict(sorted(points[point_idx].items(), key=str)).values())
            points = np.array(points, dtype=np.float64)
            # TODO: Real temperature support
            stability = calculate(dbf, comps, [phase_name], output=data['output'][:-5],
                                    T=temps, P=pressures, points=points,
                                    model=mod_srf)
            response_data -= stability[data['output'][:-5]].values.squeeze()

        response_data += np.array(data['values'], dtype=np.float64)
        response_data = response_data.flatten()
        ref = data.get('reference', '')
        dataplot_kwargs.setdefault('markersize', 8)
        dataplot_kwargs.setdefault('linestyle', 'none')
        dataplot_kwargs.setdefault('clip_on', False)
        # Cannot use setdefault because it won't overwrite previous iterations
        dataplot_kwargs['label'] = symbol_map[ref]['formatted']
        dataplot_kwargs['marker'] = symbol_map[ref]['markers']['marker']
        dataplot_kwargs['fillstyle'] = symbol_map[ref]['markers']['fillstyle']
        ax.plot(indep_var_data, response_data, **dataplot_kwargs)
    ax.set_xlim((0, 1))
    ax.set_xlabel(str(':'.join(endpoints[0])) + ' to ' + str(':'.join(endpoints[1])))
    ax.set_ylabel(plot_mapping.get(output, output))
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # legend outside
    leg.get_frame().set_edgecolor('black')
    return ax


def plot_endmember(dbf, comps, phase_name, configuration, output, datasets=None, symmetry=None, x='T', ax=None, plot_kwargs=None, dataplot_kwargs=None) -> plt.Axes:
    """
    Return one set of plotted Axes with data compared to calculated parameters

    Parameters
    ----------
    dbf : Database
        pycalphad thermodynamic database containing the relevant parameters.
    comps : Sequence[str]
        Names of components to consider in the calculation.
    phase_name : str
        Name of the considered phase phase
    configuration : Tuple[Tuple[str]]
        ESPEI-style configuration
    output : str
        Model property to plot on the y-axis e.g. ``'HM_MIX'``, or ``'SM_MIX'``.
        Must be a ``'_MIX'`` property.
    datasets : tinydb.TinyDB
    symmetry : list
        List of lists containing indices of symmetric sublattices e.g. [[0, 1], [2, 3]]
    ax : plt.Axes
        Default axes used if not specified.
    plot_kwargs : Optional[Dict[str, Any]]
        Keyword arguments to ``ax.plot`` for the predicted data.
    dataplot_kwargs : Optional[Dict[str, Any]]
        Keyword arguments to ``ax.plot`` the observed data.

    Returns
    -------
    plt.Axes

    """
    if output.endswith('_MIX'):
        raise ValueError("`plot_interaction` only supports HM, HM_FORM, SM, SM_FORM or CPM, CPM_FORM outputs.")
    if x not in ('T',):
        raise ValueError(f'`x` passed to `plot_endmember` must be "T" got {x}')
    if not plot_kwargs:
        plot_kwargs = {}
    if not dataplot_kwargs:
        dataplot_kwargs = {}

    if not ax:
        ax = plt.subplot()

    if datasets is not None:
        solver_qry = (tinydb.where('solver').test(symmetry_filter, configuration, recursive_tuplify(symmetry) if symmetry else symmetry))
        desired_data = get_prop_data(comps, phase_name, output, datasets, additional_query=solver_qry)
        desired_data = filter_configurations(desired_data, configuration, symmetry)
        desired_data = filter_temperatures(desired_data)
    else:
        desired_data = []

    # Plot predicted values from the database
    endpoints = endmembers_from_interaction(configuration)
    if len(endpoints) != 1:
        raise ValueError(f"The configuration passed to `plot_endmember` must be an endmebmer configuration. Got {configuration}")
    if output.endswith('_FORM'):
        # TODO: better reference state handling
        mod = Model(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
        prop = output[:-5]
    else:
        mod = Model(dbf, comps, phase_name)
        prop = output
    endmember = _translate_endmember_to_array(endpoints[0], mod.ast.atoms(v.SiteFraction))
    # Set up the domain of the calculation
    species = unpack_species(dbf, comps)
    # phase constituents are Species objects, so we need to be doing intersections with those
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active
    constituents = [[sp.name for sp in sorted(subl_constituents.intersection(species))] for subl_constituents in phase_constituents]
    calculate_dict = get_prop_samples(desired_data, constituents)
    potential_values = np.asarray(calculate_dict[x] if len(calculate_dict[x]) > 0 else 298.15)
    potential_grid = np.linspace(max(potential_values.min()-1, 0), potential_values.max()+1, num=100)
    predicted_values = calculate(dbf, comps, [phase_name], output=prop, T=potential_grid, P=101325, points=endmember, model=mod)[prop].values.flatten()
    ax.plot(potential_grid, predicted_values, **plot_kwargs)

    # Plot observed values
    # TODO: model exclusions handling
    bib_reference_keys = sorted({entry.get('reference', '') for entry in desired_data})
    symbol_map = bib_marker_map(bib_reference_keys)
    for data in desired_data:
        indep_var_data = None
        response_data = np.zeros_like(data['values'], dtype=np.float64)
        indep_var_data = np.array(data['conditions'][x], dtype=np.float64).flatten()

        response_data += np.array(data['values'], dtype=np.float64)
        response_data = response_data.flatten()
        ref = data.get('reference', '')
        dataplot_kwargs.setdefault('markersize', 8)
        dataplot_kwargs.setdefault('linestyle', 'none')
        dataplot_kwargs.setdefault('clip_on', False)
        # Cannot use setdefault because it won't overwrite previous iterations
        dataplot_kwargs['label'] = symbol_map[ref]['formatted']
        dataplot_kwargs['marker'] = symbol_map[ref]['markers']['marker']
        dataplot_kwargs['fillstyle'] = symbol_map[ref]['markers']['fillstyle']
        ax.plot(indep_var_data, response_data, **dataplot_kwargs)

    ax.set_xlabel(plot_mapping.get(x, x))
    ax.set_ylabel(plot_mapping.get(output, output))
    leg = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))  # legend outside
    leg.get_frame().set_edgecolor('black')
    return ax


def _translate_endmember_to_array(endmember, variables):
    site_fractions = sorted(variables, key=str)
    frac_array = np.zeros(len(site_fractions))
    for idx, component in enumerate(endmember):
        frac_array[site_fractions.index(v.SiteFraction(site_fractions[0].phase_name, idx, component))] = 1
    return frac_array
