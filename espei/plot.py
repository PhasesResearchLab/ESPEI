"""
Plotting of input data and calculated database quantities
"""
import warnings
from collections import OrderedDict

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import tinydb
from pycalphad import Model, calculate, equilibrium, variables as v
from pycalphad.core.utils import unpack_components
from pycalphad.plot.utils import phase_legend
from pycalphad.plot.eqplot import eqplot, _map_coord_to_variable, unpack_condition

from espei.error_functions.non_equilibrium_thermochemical_error import get_prop_samples
from espei.utils import bib_marker_map
from espei.core_utils import get_data, ravel_zpf_values
from espei.parameter_selection.utils import _get_sample_condition_dicts
from espei.sublattice_tools import recursive_tuplify, endmembers_from_interaction
from espei.utils import build_sitefractions

plot_mapping = {
    'T': 'Temperature (K)',
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


def plot_parameters(dbf, comps, phase_name, configuration, symmetry, datasets=None, fig=None, require_data=True):
    """
    Plot parameters of interest compared with data in subplots of a single figure

    Parameters
    ----------
    dbf : Database
        pycalphad thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phase_name : str
        Name of the considered phase phase
    configuration : tuple
        Sublattice configuration to plot, such as ('CU', 'CU') or (('CU', 'MG'), 'CU')
    symmetry : list
        List of lists containing indices of symmetric sublattices e.g. [[0, 1], [2, 3]]
    datasets : PickleableTinyDB
        ESPEI datasets to compare against. If None, nothing is plotted.
    fig : matplotlib.Figure
        Figure to create with axes as subplots.
    require_data : bool
        If True, plot parameters that have data corresponding data. Defaults to
        True. Will raise an error for non-interaction configurations.

    Returns
    -------
    None

    Examples
    --------
    >>> # plot the LAVES_C15 (Cu)(Mg) endmember
    >>> plot_parameters(dbf, ['CU', 'MG'], 'LAVES_C15', ('CU', 'MG'), symmetry=None, datasets=datasets)  # doctest: +SKIP
    >>> # plot the mixing interaction in the first sublattice
    >>> plot_parameters(dbf, ['CU', 'MG'], 'LAVES_C15', (('CU', 'MG'), 'MG'), symmetry=None, datasets=datasets)  # doctest: +SKIP

    """
    em_plots = [('T', 'CPM'), ('T', 'CPM_FORM'), ('T', 'SM'), ('T', 'SM_FORM'),
                ('T', 'HM'), ('T', 'HM_FORM')]
    mix_plots = [ ('Z', 'HM_MIX'), ('Z', 'SM_MIX')]
    comps = sorted(comps)
    mod = Model(dbf, comps, phase_name)
    mod.models['idmix'] = 0
    # This is for computing properties of formation
    mod_norefstate = Model(dbf, comps, phase_name, parameters={'GHSER'+(c.upper()*2)[:2]: 0 for c in comps})
    # Is this an interaction parameter or endmember?
    if any([isinstance(conf, list) or isinstance(conf, tuple) for conf in configuration]):
        plots = mix_plots
    else:
        plots = em_plots

    # filter which parameters to plot by the data that exists
    if require_data and datasets is not None:
        filtered_plots = []
        for x_val, y_val in plots:
            desired_props = [y_val.split('_')[0]+'_FORM', y_val] if y_val.endswith('_MIX') else [y_val]
            data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
            if len(data) > 0:
                filtered_plots.append((x_val, y_val, data))
    elif require_data:
        raise ValueError('Plots require datasets, but no datasets were passed.')
    elif plots == em_plots and not require_data:
        # How we treat temperature dependence is ambiguous when there is no data, so we raise an error
        raise ValueError('The "require_data=False" option is not supported for non-mixing configurations.')
    elif datasets is not None:
        filtered_plots = []
        for x_val, y_val in plots:
            desired_props = [y_val.split('_')[0]+'_FORM', y_val] if y_val.endswith('_MIX') else [y_val]
            data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
            filtered_plots.append((x_val, y_val, data))
    else:
        filtered_plots = [(x_val, y_val, []) for x_val, y_val in plots]

    num_plots = len(filtered_plots)
    if num_plots == 0:
        return
    if not fig:
        fig = plt.figure(figsize=plt.figaspect(num_plots))

    # plot them
    for i, (x_val, y_val, data) in enumerate(filtered_plots):
        if y_val.endswith('_FORM'):
            ax = fig.add_subplot(num_plots, 1, i+1)
            ax = _compare_data_to_parameters(dbf, comps, phase_name, data, mod_norefstate, configuration, x_val, y_val, ax=ax)
        else:
            ax = fig.add_subplot(num_plots, 1, i+1)
            ax = _compare_data_to_parameters(dbf, comps, phase_name, data, mod, configuration, x_val, y_val, ax=ax)


def dataplot(comps, phases, conds, datasets, tielines=True, ax=None, plot_kwargs=None, tieline_plot_kwargs=None):
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
        ax = plt.gca(projection=projection)
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
    bib_reference_keys = sorted(list({entry['reference'] for entry in desired_data}))
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
    legend_handles, phase_color_map = phase_legend(phases)

    if projection is None:
        # TODO: There are lot of ways this could break in multi-component situations

        # plot x vs. T
        y = 'T'

        # handle plotting kwargs
        scatter_kwargs = {'markersize': 6, 'markeredgewidth': 1}
        # raise warnings if any of the aliased versions of the default values are used
        possible_aliases = [('markersize', 'ms'), ('markeredgewidth', 'mew')]
        for actual_arg, aliased_arg in possible_aliases:
            if aliased_arg in plot_kwargs:
                warnings.warn("'{0}' passed as plotting keyword argument to dataplot, but the alias '{1}' is already set to '{2}'. Use the full version of the keyword argument '{1}' to override the default.".format(aliased_arg, actual_arg, scatter_kwargs.get(actual_arg)))
        scatter_kwargs.update(plot_kwargs)

        eq_dict = ravel_zpf_values(desired_data, [x])

        # two phase
        updated_tieline_plot_kwargs = {'linewidth':1, 'color':'k'}
        if tieline_plot_kwargs is not None:
            updated_tieline_plot_kwargs.update(tieline_plot_kwargs)
        equilibria_to_plot = eq_dict.get(1, [])
        equilibria_to_plot.extend(eq_dict.get(2, []))
        equilibria_to_plot.extend(eq_dict.get(3, []))
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

        eq_dict = ravel_zpf_values(desired_data, [x, y], {'T': conds[v.T]})

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


def eqdataplot(eq, datasets, ax=None, plot_kwargs=None):
    """
    Plot datapoints corresponding to the components and phases in the eq Dataset.
    A convenience function for dataplot.

    Parameters
    ----------
    eq : xarray.Dataset
        Result of equilibrium calculation.
    datasets : PickleableTinyDB
        Database of phase equilibria datasets
    ax : matplotlib.Axes
        Default axes used if not specified.
    plot_kwargs : dict
        Keyword arguments to pass to dataplot

    Returns
    -------
    A plot of phase equilibria points as a figure

    Examples
    --------

    >>> from pycalphad import equilibrium, Database, variables as v  # doctest: +SKIP
    >>> from pycalphad.plot.eqplot import eqplot  # doctest: +SKIP
    >>> from espei.datasets import load_datasets, recursive_glob  # doctest: +SKIP
    >>> datasets = load_datasets(recursive_glob('.', '*.json'))  # doctest: +SKIP
    >>> dbf = Database('my_databases.tdb')  # doctest: +SKIP
    >>> my_phases = list(dbf.phases.keys())  # doctest: +SKIP
    >>> eq = equilibrium(dbf, ['CU', 'MG', 'VA'], my_phases, {v.P: 101325, v.T: (500, 1000, 10), v.X('MG'): (0, 1, 0.01)})  # doctest: +SKIP
    >>> ax = eqplot(eq)  # doctest: +SKIP
    >>> ax = eqdataplot(eq, datasets, ax=ax)  # doctest: +SKIP

    """
    deprecation_msg = (
        "`espei.plot.eqdataplot` is deprecated and will be removed in ESPEI 0.9. "
        "Users depending on plotting from an `pycalphad.equilibrium` result should use "
        "`pycalphad.plot.eqplot.eqplot` along with `espei.plot.dataplot` directly. "
        "Note that pycalphad's mapping can offer signficant reductions in calculation "
        "time compared to using `equilibrium` followed by `eqplot`."
    )
    warnings.warn(deprecation_msg, category=FutureWarning)
    # TODO: support reference legend
    conds = OrderedDict([(_map_coord_to_variable(key), unpack_condition(np.asarray(value)))
                         for key, value in sorted(eq.coords.items(), key=str)
                         if (key == 'T') or (key == 'P') or (key.startswith('X_'))])

    phases = list(map(str, sorted(set(np.array(eq.Phase.values.ravel(), dtype='U')) - {''}, key=str)))
    comps = list(map(str, sorted(np.array(eq.coords['component'].values, dtype='U'), key=str)))

    ax = dataplot(comps, phases, conds, datasets, ax=ax, plot_kwargs=plot_kwargs)

    return ax


def multiplot(dbf, comps, phases, conds, datasets, eq_kwargs=None, plot_kwargs=None, data_kwargs=None):
    """
    Plot a phase diagram with datapoints described by datasets.
    This is a wrapper around pycalphad.equilibrium, pycalphad's eqplot, and dataplot.

    Parameters
    ----------
    dbf : Database
        pycalphad thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phases : list
        Names of phases to consider in the calculation.
    conds : dict
        Maps StateVariables to values and/or iterables of values.
    datasets : PickleableTinyDB
        Database of phase equilibria datasets
    eq_kwargs : dict
        Keyword arguments passed to pycalphad equilibrium()
    plot_kwargs : dict
        Keyword arguments passed to pycalphad eqplot()
    data_kwargs : dict
        Keyword arguments passed to dataplot()

    Returns
    -------
    A phase diagram with phase equilibria data as a figure

    Examples
    --------

    >>> from pycalphad import Database, variables as v  # doctest: +SKIP
    >>> from pycalphad.plot.eqplot import eqplot  # doctest: +SKIP
    >>> from espei.datasets import load_datasets, recursive_glob  # doctest: +SKIP
    >>> datasets = load_datasets(recursive_glob('.', '*.json'))  # doctest: +SKIP
    >>> dbf = Database('my_databases.tdb')  # doctest: +SKIP
    >>> my_phases = list(dbf.phases.keys())  # doctest: +SKIP
    >>> multiplot(dbf, ['CU', 'MG', 'VA'], my_phases, {v.P: 101325, v.T: 1000, v.X('MG'): (0, 1, 0.01)}, datasets)  # doctest: +SKIP

    """
    deprecation_msg = (
        "`espei.plot.multiplot` is deprecated and will be removed in ESPEI 0.9. "
        "Users depending on `multiplot` should use pycalphad's `binplot` or `ternplot` "
        "followed by `espei.plot.dataplot`. Note that pycalphad's mapping can offer "
        "signficant reductions in calculation time compared to using `multiplot`. See "
        "ESPEI's recipes for an example: "
        "https://espei.org/en/latest/recipes.html#plot-phase-diagram-with-data"
    )
    warnings.warn(deprecation_msg, category=FutureWarning)
    eq_kwargs = eq_kwargs or dict()
    plot_kwargs = plot_kwargs or dict()
    data_kwargs = data_kwargs or dict()

    eq_result = equilibrium(dbf, comps, phases, conds, **eq_kwargs)
    ax = eqplot(eq_result, **plot_kwargs)
    ax = eqdataplot(eq_result, datasets, ax=ax, plot_kwargs=data_kwargs)
    return ax


def _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod, configuration, x, y, ax=None):
    """
    Return one set of plotted Axes with data compared to calculated parameters

    Parameters
    ----------
    dbf : Database
        pycalphad thermodynamic database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phase_name : str
        Name of the considered phase phase
    desired_data :
    mod : Model
        A pycalphad Model. The Model may or may not have the reference state zeroed out for formation properties.
    configuration :
    x : str
        Model property to plot on the x-axis e.g. 'T', 'HM_MIX', 'SM_FORM'
    y : str
        Model property to plot on the y-axis e.g. 'T', 'HM_MIX', 'SM_FORM'
    ax : matplotlib.Axes
        Default axes used if not specified.

    Returns
    -------
    matplotlib.Axes

    """
    species = unpack_components(dbf, comps)
    # phase constituents are Species objects, so we need to be doing intersections with those
    phase_constituents = dbf.phases[phase_name].constituents
    # phase constituents must be filtered to only active:
    constituents = [[sp.name for sp in sorted(subl_constituents.intersection(species))] for subl_constituents in phase_constituents]
    subl_dof = list(map(len, constituents))
    calculate_dict = get_prop_samples(desired_data, constituents)
    sample_condition_dicts = _get_sample_condition_dicts(calculate_dict, subl_dof)
    endpoints = endmembers_from_interaction(configuration)
    interacting_subls = [c for c in recursive_tuplify(configuration) if isinstance(c, tuple)]
    disordered_config = False
    if (len(set(interacting_subls)) == 1) and (len(interacting_subls[0]) == 2):
        # This configuration describes all sublattices with the same two elements interacting
        # In general this is a high-dimensional space; just plot the diagonal to see the disordered mixing
        endpoints = [endpoints[0], endpoints[-1]]
        disordered_config = True
    if not ax:
        fig = plt.figure(figsize=plt.figaspect(1))
        ax = fig.gca()
    bar_chart = False
    bar_labels = []
    bar_data = []
    if y.endswith('_FORM'):
        # We were passed a Model object with zeroed out reference states
        yattr = y[:-5]
    else:
        yattr = y
    if len(endpoints) == 1:
        # This is an endmember so we can just compute T-dependent stuff
        Ts = calculate_dict['T']
        temperatures = np.asarray(Ts if len(Ts) > 0 else 298.15)
        if temperatures.min() != temperatures.max():
            temperatures = np.linspace(temperatures.min(), temperatures.max(), num=100)
        else:
            # We only have one temperature: let's do a bar chart instead
            bar_chart = True
            temperatures = temperatures.min()
        endmember = _translate_endmember_to_array(endpoints[0], mod.ast.atoms(v.SiteFraction))[None, None]
        predicted_quantities = calculate(dbf, comps, [phase_name], output=yattr,
                                         T=temperatures, P=101325, points=endmember, model=mod, mode='numpy')
        if y == 'HM' and x == 'T':
            # Shift enthalpy data so that value at minimum T is zero
            predicted_quantities[yattr] -= predicted_quantities[yattr].sel(T=temperatures[0]).values.flatten()
        response_data = predicted_quantities[yattr].values.flatten()
        if not bar_chart:
            extra_kwargs = {}
            if len(response_data) < 10:
                extra_kwargs['markersize'] = 20
                extra_kwargs['marker'] = '.'
                extra_kwargs['linestyle'] = 'none'
                extra_kwargs['clip_on'] = False
            ax.plot(temperatures, response_data,
                           label='This work', color='k', **extra_kwargs)
            ax.set_xlabel(plot_mapping.get(x, x))
            ax.set_ylabel(plot_mapping.get(y, y))
        else:
            bar_labels.append('This work')
            bar_data.append(response_data[0])
    elif len(endpoints) == 2:
        # Binary interaction parameter
        first_endpoint = _translate_endmember_to_array(endpoints[0], mod.ast.atoms(v.SiteFraction))
        second_endpoint = _translate_endmember_to_array(endpoints[1], mod.ast.atoms(v.SiteFraction))
        point_matrix = np.linspace(0, 1, num=100)[None].T * second_endpoint + \
            (1 - np.linspace(0, 1, num=100))[None].T * first_endpoint
        # TODO: Real temperature support
        point_matrix = point_matrix[None, None]
        predicted_quantities = calculate(dbf, comps, [phase_name], output=yattr,
                                         T=300, P=101325, points=point_matrix, model=mod, mode='numpy')
        response_data = predicted_quantities[yattr].values.flatten()
        if not bar_chart:
            extra_kwargs = {}
            if len(response_data) < 10:
                extra_kwargs['markersize'] = 20
                extra_kwargs['marker'] = '.'
                extra_kwargs['linestyle'] = 'none'
                extra_kwargs['clip_on'] = False
            ax.plot(np.linspace(0, 1, num=100), response_data, label='This work', color='k', **extra_kwargs)
            ax.set_xlim((0, 1))
            ax.set_xlabel(str(':'.join(endpoints[0])) + ' to ' + str(':'.join(endpoints[1])))
            ax.set_ylabel(plot_mapping.get(y, y))
        else:
            bar_labels.append('This work')
            bar_data.append(response_data[0])
    else:
        raise NotImplementedError('No support for plotting configuration {}'.format(configuration))

    bib_reference_keys = sorted(list({entry['reference'] for entry in desired_data}))
    symbol_map = bib_marker_map(bib_reference_keys)

    for data in desired_data:
        indep_var_data = None
        response_data = np.zeros_like(data['values'], dtype=np.float_)
        if x == 'T' or x == 'P':
            indep_var_data = np.array(data['conditions'][x], dtype=np.float_).flatten()
        elif x == 'Z':
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
            if y.endswith('_MIX') and data['output'].endswith('_FORM'):
                # All the _FORM data we have still has the lattice stability contribution
                # Need to zero it out to shift formation data to mixing
                mod_latticeonly = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
                mod_latticeonly.models = {key: value for key, value in mod_latticeonly.models.items()
                                          if key == 'ref'}
                temps = data['conditions'].get('T', 300)
                pressures = data['conditions'].get('P', 101325)
                points = build_sitefractions(phase_name, data['solver']['sublattice_configurations'],
                                             data['solver']['sublattice_occupancies'])
                for point_idx in range(len(points)):
                    missing_variables = mod_latticeonly.ast.atoms(v.SiteFraction) - set(points[point_idx].keys())
                    # Set unoccupied values to zero
                    points[point_idx].update({key: 0 for key in missing_variables})
                    # Change entry to a sorted array of site fractions
                    points[point_idx] = list(OrderedDict(sorted(points[point_idx].items(), key=str)).values())
                points = np.array(points, dtype=np.float_)
                # TODO: Real temperature support
                points = points[None, None]
                stability = calculate(dbf, comps, [phase_name], output=data['output'][:-5],
                                      T=temps, P=pressures, points=points,
                                      model=mod_latticeonly, mode='numpy')
                response_data -= stability[data['output'][:-5]].values.squeeze()

        response_data += np.array(data['values'], dtype=np.float_)
        response_data = response_data.flatten()
        if not bar_chart:
            extra_kwargs = {}
            extra_kwargs['markersize'] = 8
            extra_kwargs['linestyle'] = 'none'
            extra_kwargs['clip_on'] = False
            ref = data.get('reference', '')
            mark = symbol_map[ref]['markers']
            ax.plot(indep_var_data, response_data,
                    label=symbol_map[ref]['formatted'],
                    marker=mark['marker'],
                    fillstyle=mark['fillstyle'],
                    **extra_kwargs)
        else:
            bar_labels.append(data.get('reference', None))
            bar_data.append(response_data[0])
    if bar_chart:
        ax.barh(0.02 * np.arange(len(bar_data)), bar_data,
                       color='k', height=0.01)
        endmember_title = ' to '.join([':'.join(i) for i in endpoints])
        ax.get_figure().suptitle('{} (T = {} K)'.format(endmember_title, temperatures), fontsize=20)
        ax.set_yticks(0.02 * np.arange(len(bar_data)))
        ax.set_yticklabels(bar_labels, fontsize=20)
        # This bar chart is rotated 90 degrees, so "y" is now x
        ax.set_xlabel(plot_mapping.get(y, y))
    else:
        ax.set_frame_on(False)
        leg = ax.legend(loc='best')
        leg.get_frame().set_edgecolor('black')
    return ax


def _translate_endmember_to_array(endmember, variables):
    site_fractions = sorted(variables, key=str)
    frac_array = np.zeros(len(site_fractions))
    for idx, component in enumerate(endmember):
        frac_array[site_fractions.index(v.SiteFraction(site_fractions[0].phase_name, idx, component))] = 1
    return frac_array
