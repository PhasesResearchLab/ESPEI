"""
Plotting of input data and calculated database quantities
"""
import numpy as np
import tinydb
from collections import OrderedDict
from pycalphad import Model, calculate, variables as v
from pycalphad.plot.utils import phase_legend

from espei.core_utils import get_data, get_samples, list_to_tuple, \
    endmembers_from_interaction, build_sitefractions

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


def plot_parameters(dbf, comps, phase_name, configuration, symmetry, datasets=None):
    em_plots = [('T', 'CPM'), ('T', 'CPM_FORM'), ('T', 'SM'), ('T', 'SM_FORM'),
                ('T', 'HM'), ('T', 'HM_FORM')]
    mix_plots = [('Z', 'HM_FORM'), ('Z', 'HM_MIX'), ('Z', 'SM_MIX')]
    comps = sorted(comps)
    mod = Model(dbf, comps, phase_name)
    # This is for computing properties of formation
    mod_norefstate = Model(dbf, comps, phase_name, parameters={'GHSER'+c.upper(): 0 for c in comps})
    # Is this an interaction parameter or endmember?
    if any([isinstance(conf, list) or isinstance(conf, tuple) for conf in configuration]):
        plots = mix_plots
    else:
        plots = em_plots
    for x_val, y_val in plots:
        if datasets is not None:
            if y_val.endswith('_MIX'):
                desired_props = [y_val.split('_')[0]+'_FORM', y_val]
            else:
                desired_props = [y_val]
            desired_data = get_data(comps, phase_name, configuration, symmetry, datasets, desired_props)
        else:
            desired_data = []
        if len(desired_data) == 0:
            continue
        if y_val.endswith('_FORM'):
            _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod_norefstate, configuration, x_val, y_val)
        else:
            _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod, configuration, x_val, y_val)


def multi_plot(dbf, comps, phases, datasets, ax=None):
    import matplotlib.pyplot as plt
    plots = [('ZPF', 'T')]
    real_components = sorted(set(comps) - {'VA'})
    legend_handles, phase_color_map = phase_legend(phases)
    for output, indep_var in plots:
        desired_data = datasets.search((tinydb.where('output') == output) &
                                       (tinydb.where('components').test(lambda x: set(x).issubset(comps))) &
                                       (tinydb.where('phases').test(lambda x: len(set(phases).intersection(x)) > 0)))
        ax = ax if ax is not None else plt.gca()
        # TODO: There are lot of ways this could break in multi-component situations
        chosen_comp = real_components[-1]
        ax.set_xlabel('X({})'.format(chosen_comp))
        ax.set_ylabel(indep_var)
        ax.set_xlim((0, 1))
        symbol_map = {1: "o", 2: "s", 3: "^"}
        for data in desired_data:
            payload = data['values']
            # TODO: Add broadcast_conditions support
            # Repeat the temperature (or whatever variable) vector to align with the unraveled data
            temp_repeats = np.zeros(len(np.atleast_1d(data['conditions'][indep_var])), dtype=np.int)
            for idx, x in enumerate(payload):
                temp_repeats[idx] = len(x)
            temps_ravelled = np.repeat(data['conditions'][indep_var], temp_repeats)
            payload_ravelled = []
            phases_ravelled = []
            comps_ravelled = []
            symbols_ravelled = []
            # TODO: Fix to only include equilibria listed in 'phases'
            for p in payload:
                symbols_ravelled.extend([symbol_map[len(p)]] * len(p))
                payload_ravelled.extend(p)
            for rp in payload_ravelled:
                phases_ravelled.append(rp[0])
                comp_dict = dict(zip([x.upper() for x in rp[1]], rp[2]))
                dependent_comp = list(set(real_components) - set(comp_dict.keys()))
                if len(dependent_comp) > 1:
                    raise ValueError('Dependent components greater than one')
                elif len(dependent_comp) == 1:
                    dependent_comp = dependent_comp[0]
                    # TODO: Assuming N=1
                    comp_dict[dependent_comp] = 1 - sum(np.array(list(comp_dict.values()), dtype=np.float))
                chosen_comp_value = comp_dict[chosen_comp]
                comps_ravelled.append(chosen_comp_value)
            symbols_ravelled = np.array(symbols_ravelled)
            comps_ravelled = np.array(comps_ravelled)
            temps_ravelled = np.array(temps_ravelled)
            phases_ravelled = np.array(phases_ravelled)
            # We can't pass an array of markers to scatter, sadly
            for sym in symbols_ravelled:
                selected = symbols_ravelled == sym
                ax.scatter(comps_ravelled[selected], temps_ravelled[selected], marker=sym, s=100,
                           c='none', edgecolors=[phase_color_map[x] for x in phases_ravelled[selected]])
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5))


def _compare_data_to_parameters(dbf, comps, phase_name, desired_data, mod, configuration, x, y):
    import matplotlib.pyplot as plt
    all_samples = np.array(get_samples(desired_data), dtype=np.object)
    endpoints = endmembers_from_interaction(configuration)
    interacting_subls = [c for c in list_to_tuple(configuration) if isinstance(c, tuple)]
    disordered_config = False
    if (len(set(interacting_subls)) == 1) and (len(interacting_subls[0]) == 2):
        # This configuration describes all sublattices with the same two elements interacting
        # In general this is a high-dimensional space; just plot the diagonal to see the disordered mixing
        endpoints = [endpoints[0], endpoints[-1]]
        disordered_config = True
    fig = plt.figure(figsize=(9, 9))
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
        temperatures = np.array([i[0] for i in all_samples], dtype=np.float)
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
            fig.gca().plot(temperatures, response_data,
                           label='This work', color='k', **extra_kwargs)
            fig.gca().set_xlabel(plot_mapping.get(x, x))
            fig.gca().set_ylabel(plot_mapping.get(y, y))
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
            fig.gca().plot(np.linspace(0, 1, num=100), response_data,
                           label='This work', color='k', **extra_kwargs)
            fig.gca().set_xlim((0, 1))
            fig.gca().set_xlabel(str(':'.join(endpoints[0])) + ' to ' + str(':'.join(endpoints[1])))
            fig.gca().set_ylabel(plot_mapping.get(y, y))
        else:
            bar_labels.append('This work')
            bar_data.append(response_data[0])
    else:
        raise NotImplementedError('No support for plotting configuration {}'.format(configuration))

    for data in desired_data:
        indep_var_data = None
        response_data = np.zeros_like(data['values'], dtype=np.float)
        if x == 'T' or x == 'P':
            indep_var_data = np.array(data['conditions'][x], dtype=np.float).flatten()
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
                interactions = np.array([i[1][1] for i in get_samples([data])], dtype=np.float)
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
                points = np.array(points, dtype=np.float)
                # TODO: Real temperature support
                points = points[None, None]
                stability = calculate(dbf, comps, [phase_name], output=data['output'][:-5],
                                      T=temps, P=pressures, points=points,
                                      model=mod_latticeonly, mode='numpy')
                response_data -= stability[data['output'][:-5]].values

        response_data += np.array(data['values'], dtype=np.float)
        response_data = response_data.flatten()
        if not bar_chart:
            extra_kwargs = {}
            if len(response_data) < 10:
                extra_kwargs['markersize'] = 20
                extra_kwargs['marker'] = '.'
                extra_kwargs['linestyle'] = 'none'
                extra_kwargs['clip_on'] = False

            fig.gca().plot(indep_var_data, response_data, label=data.get('reference', None),
                           **extra_kwargs)
        else:
            bar_labels.append(data.get('reference', None))
            bar_data.append(response_data[0])
    if bar_chart:
        fig.gca().barh(0.02 * np.arange(len(bar_data)), bar_data,
                       color='k', height=0.01)
        endmember_title = ' to '.join([':'.join(i) for i in endpoints])
        fig.suptitle('{} (T = {} K)'.format(endmember_title, temperatures), fontsize=20)
        fig.gca().set_yticks(0.02 * np.arange(len(bar_data)))
        fig.gca().set_yticklabels(bar_labels, fontsize=20)
        # This bar chart is rotated 90 degrees, so "y" is now x
        fig.gca().set_xlabel(plot_mapping.get(y, y))
    else:
        fig.gca().set_frame_on(False)
        leg = fig.gca().legend(loc='best')
        leg.get_frame().set_edgecolor('black')
    fig.canvas.draw()


def _translate_endmember_to_array(endmember, variables):
    site_fractions = sorted(variables, key=str)
    frac_array = np.zeros(len(site_fractions))
    for idx, component in enumerate(endmember):
        frac_array[site_fractions.index(v.SiteFraction(site_fractions[0].phase_name, idx, component))] = 1
    return frac_array
