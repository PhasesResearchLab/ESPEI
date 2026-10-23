"""
Generate the invented A-B-C and A-B-C-D datasets from abcd.tdb at its true parameter values.

Values come from pycalphad's equilibrium and symbolic Model properties, not from ESPEI's residual code.
Run ``python generate.py`` to write the JSON files, or ``python generate.py --check`` to compare with them.
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from pycalphad import Database, Model, equilibrium, variables as v

HERE = os.path.dirname(os.path.abspath(__file__))
P = 101325.0
REFERENCE = "invented (tests/gradient_data/generate.py)"
SYSTEMS = {'ternary': ['A', 'B', 'C'], 'quaternary': ['A', 'B', 'C', 'D']}

# (T, overall composition, stable phases) of each ZPF region; compositions are of the independent elements
ZPF_REGIONS = {
    'ternary': {
        'two_phase': [(900, (0.05, 0.15), ['FCC_A1', 'TAU']), (900, (0.15, 0.45), ['ORD', 'TAU']), (1150, (0.45, 0.35), ['LIQUID', 'ORD'])],
        'three_phase': [(900, (0.25, 0.25), ['FCC_A1', 'ORD', 'TAU'])],
        'null': [(900, (0.15, 0.15), ['FCC_A1', 'TAU']), (1150, (0.35, 0.45), ['LIQUID', 'ORD'])],
        'disordered': [(900, (0.35, 0.45), ['ORD', 'TAU']), (1150, (0.45, 0.35), ['LIQUID', 'ORD'])],
        'hyperplane': [(900, (0.05, 0.25), ['FCC_A1', 'TAU']), (900, (0.15, 0.25), ['FCC_A1', 'TAU'])],
        'stoichiometric': [(900, (0.35, 0.05), ['AB2', 'FCC_A1']), (900, (0.55, 0.05), ['AB2', 'FCC_A1', 'ORD'])],
    },
    'quaternary': {
        'two_phase': [(900, (0.05, 0.15, 0.1), ['FCC_A1', 'TAU']), (900, (0.25, 0.45, 0.1), ['ORD', 'TAU']), (1150, (0.35, 0.35, 0.1), ['LIQUID', 'ORD'])],
        'three_phase': [(900, (0.35, 0.15, 0.1), ['FCC_A1', 'ORD', 'TAU'])],
        'null': [(900, (0.15, 0.15, 0.1), ['FCC_A1', 'TAU']), (1150, (0.25, 0.45, 0.1), ['LIQUID', 'ORD'])],
        'disordered': [(900, (0.25, 0.45, 0.1), ['ORD', 'TAU']), (1150, (0.35, 0.35, 0.1), ['LIQUID', 'ORD'])],
        'hyperplane': [(900, (0.05, 0.25, 0.1), ['FCC_A1', 'TAU']), (900, (0.15, 0.25, 0.1), ['FCC_A1', 'TAU'])],
        'stoichiometric': [(900, (0.45, 0.05, 0.1), ['AB2', 'FCC_A1'])],
    },
}
# A-B edge: the AB2 vertex's own equilibrium is single-phase without internal degrees of freedom
EDGE_REGION = (900, (0.5,), ['AB2', 'FCC_A1'])
LIQUID_COMPOSITIONS = {
    'ternary': [(0.15, 0.05), (0.25, 0.15), (0.35, 0.05), (0.05, 0.25)],
    'quaternary': [(0.15, 0.05, 0.1), (0.25, 0.15, 0.1), (0.35, 0.05, 0.1), (0.05, 0.25, 0.1)],
}
FCC_TAU_COMPOSITIONS = {
    'ternary': [(0.05, 0.15), (0.15, 0.25)],
    'quaternary': [(0.05, 0.15, 0.1), (0.15, 0.25, 0.1)],
}
# (site ratios, configurations, occupancies) of the fixed-configuration data. ESPEI's fixed-configuration Workspaces fail
# when the last element of the Database (D) and another element are both absent, so ternary configurations contain A, B and C.
FIXED_CONFIGURATIONS = {
    'ternary': {
        'TAU': ([1, 1], [[['A', 'B'], ['C', 'VA']], [['A', 'B'], 'C'], [['A', 'B'], ['C', 'VA']]],
                [[[0.7, 0.3], [0.6, 0.4]], [[0.6, 0.4], 1], [[0.3, 0.7], [0.9, 0.1]]]),
        'ORD': ([1, 1], [[['A', 'B', 'C'], ['A', 'B', 'C']], [['A', 'B', 'C'], ['A', 'B', 'C']]],
                [[[0.2, 0.4, 0.4], [0.2, 0.4, 0.4]], [[0.4, 0.3, 0.3], [0.4, 0.3, 0.3]]]),
        'LIQUID': ([1], [[['A', 'B', 'C']], [['A', 'B', 'C']]], [[[0.3, 0.4, 0.3]], [[0.5, 0.2, 0.3]]]),
    },
    'quaternary': {
        'TAU': ([1, 1], [['B', 'D'], ['A', ['C', 'D']], [['A', 'B'], ['C', 'D', 'VA']]],
                [[1, 1], [1, [0.5, 0.5]], [[0.6, 0.4], [0.5, 0.3, 0.2]]]),
        'ORD': ([1, 1], [[['B', 'C', 'D'], ['B', 'C', 'D']], [['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D']]],
                [[[0.3, 0.4, 0.3], [0.3, 0.4, 0.3]], [[0.1, 0.3, 0.4, 0.2], [0.1, 0.3, 0.4, 0.2]]]),
        'LIQUID': ([1], [[['A', 'B', 'C', 'D']], [['C', 'D']]], [[[0.25, 0.25, 0.3, 0.2]], [[0.5, 0.5]]]),
    },
}
FIXED_TEMPERATURES = [800.0, 1200.0]


def _round(x):
    return float(f'{float(x):.10g}')


def _conditions(elements, T, composition):
    return {v.T: T, v.P: P, v.N: 1, **{v.X(el): x for el, x in zip(elements[1:], composition)}}


def _equilibrium(dbf, elements, phases, T, composition, output='GM'):
    """Converged equilibrium; raises if it failed."""
    eq = equilibrium(dbf, elements + ['VA'], phases, _conditions(elements, T, composition), output=output)
    if np.any(np.isnan(eq.MU.values)):
        raise RuntimeError(f'Equilibrium at T={T}, X={composition} did not converge')
    return eq


def _stable(eq, dbf, elements):
    """(phase, composition, site fractions) of each stable composition set."""
    stable = []
    for phase, x, y in zip(eq.Phase.values.squeeze(), eq.X.values.squeeze(), eq.Y.values.squeeze()):
        if phase == '':
            continue
        num_dof = sum(len(set(subl).intersection(v.Species(el) for el in elements + ['VA'])) for subl in dbf.phases[phase].constituents)
        stable.append((str(phase), np.asarray(x), np.asarray(y)[:num_dof]))
    return stable


def zpf_region(dbf, elements, T, composition, expected, null=(), disordered=(), hyperplane=None):
    """ZPF region from the equilibrium at an overall composition, asserting its stable phases."""
    stable = _stable(_equilibrium(dbf, elements, list(dbf.phases), T, composition), dbf, elements)
    phases = sorted(phase for phase, _, _ in stable)
    if phases != sorted(expected):
        raise RuntimeError(f'Expected {sorted(expected)} at T={T}, X={composition}, got {phases}')
    region = []
    if hyperplane is not None:
        region.append(['__HYPERPLANE__', elements[1:], [_round(x) for x in composition]])
    for phase, x, y in sorted(stable, key=lambda s: s[0]):
        if phase == 'ORD' and not np.allclose(y[:len(y)//2], y[len(y)//2:], atol=1e-8):
            raise RuntimeError(f'ORD is ordered at T={T}, X={composition}: {y}')
        fractions = [None] * (len(elements) - 1) if phase in null else [_round(xi) for xi in x[1:]]
        region.append([phase, elements[1:], fractions] + (['disordered'] if phase in disordered else []))
    return region


def zpf_dataset(elements, regions, comment):
    phases = sorted({vertex[0] for _, region in regions for vertex in region} - {'__HYPERPLANE__'})
    return {
        'components': elements + ['VA'], 'phases': phases, 'output': 'ZPF',
        'conditions': {'P': P, 'T': [float(T) for T, _ in regions]},
        'values': [region for _, region in regions],
        'reference': REFERENCE, 'comment': comment,
    }


def zpf_datasets(dbf, system):
    elements = SYSTEMS[system]
    regions = ZPF_REGIONS[system]
    datasets = {}
    datasets['zpf_two_phase'] = zpf_dataset(elements, [(T, zpf_region(dbf, elements, T, x, ph)) for T, x, ph in regions['two_phase']],
                                            'Two-phase tie-lines with known compositions')
    datasets['zpf_three_phase'] = zpf_dataset(elements, [(T, zpf_region(dbf, elements, T, x, ph)) for T, x, ph in regions['three_phase']],
                                              'Three-phase region')
    datasets['zpf_null'] = zpf_dataset(elements, [(T, zpf_region(dbf, elements, T, x, ph, null=ph[1:])) for T, x, ph in regions['null']],
                                       'Tie-lines with a null-composition vertex')
    datasets['zpf_disordered'] = zpf_dataset(elements, [(T, zpf_region(dbf, elements, T, x, ph, disordered=['ORD'])) for T, x, ph in regions['disordered']],
                                             'Tie-lines with a disordered ORD vertex')
    (T1, x1, ph1), (T2, x2, ph2) = regions['hyperplane']
    datasets['zpf_hyperplane'] = zpf_dataset(elements, [(T1, zpf_region(dbf, elements, T1, x1, ph1, hyperplane=True)),
                                                        (T2, zpf_region(dbf, elements, T2, x2, ph2, null=ph2, hyperplane=True))],
                                             '__HYPERPLANE__ vertices with known and with null phase compositions')
    datasets['zpf_stoichiometric'] = zpf_dataset(elements, [(T, zpf_region(dbf, elements, T, x, ph, null=['AB2'])) for T, x, ph in regions['stoichiometric']],
                                                 'Null-composition AB2 vertices: its X of the other elements is zero, where equilibria fail')
    T, x, ph = EDGE_REGION
    edge = zpf_region(dbf, ['A', 'B'], T, x, ph)
    for vertex in edge:
        if vertex[0] == 'AB2' and not np.isclose(vertex[2][0], 2/3):
            raise RuntimeError(f'AB2 vertex is not at its composition: {vertex}')
    datasets['zpf_edge_stoichiometric'] = zpf_dataset(['A', 'B'], [(T, edge)],
                                                      'A-B tie-line with a known-composition AB2 vertex, whose own equilibrium is masked from the hyperplane')
    return datasets


def _pure_property(dbf, element, phase, prop, T):
    mod = Model(dbf, [element, 'VA'], phase)
    return float(getattr(mod, prop).xreplace({v.T: T, v.P: P, **{y: 1.0 for y in mod.site_fractions}}))


def activity_datasets(dbf, system):
    elements = SYSTEMS[system]
    T = 1150.0
    compositions = LIQUID_COMPOSITIONS[system]
    reference = _equilibrium(dbf, elements, ['LIQUID'], T, [0.0] * (len(elements) - 1))
    mu_ref = float(reference.MU.values.squeeze()[0])
    activities = []
    for composition in compositions:
        eq = _equilibrium(dbf, elements, list(dbf.phases), T, composition)
        phases = sorted(phase for phase, _, _ in _stable(eq, dbf, elements))
        if phases != ['LIQUID']:
            raise RuntimeError(f'Expected LIQUID at T={T}, X={composition}, got {phases}')
        activities.append(_round(np.exp((float(eq.MU.values.squeeze()[0]) - mu_ref) / (float(v.R) * T))))
    return {'acr_liquid': {
        'components': elements + ['VA'], 'phases': ['LIQUID'], 'output': 'ACR_A',
        'reference_state': {'phases': ['LIQUID'], 'conditions': {'P': P, 'T': T, **{f'X_{el}': 0.0 for el in elements[1:]}}},
        'conditions': {'P': P, 'T': T, **{f'X_{el}': [c[i] for c in compositions] for i, el in enumerate(elements[1:])}},
        'values': [[activities]],
        'reference': REFERENCE, 'comment': 'Liquid activity of A, referenced to pure liquid A',
    }}


def equilibrium_property_datasets(dbf, system):
    elements = SYSTEMS[system]
    datasets = {}
    cases = [('liquid', 1150.0, ['LIQUID'], LIQUID_COMPOSITIONS[system], ['HM', 'SM', 'CPM', 'HMR', 'SMR'], 'LIQUID'),
             ('fcc_tau', 900.0, ['FCC_A1', 'TAU'], FCC_TAU_COMPOSITIONS[system], ['HM', 'SM', 'CPM', 'HMR'], 'FCC_A1')]
    for name, T, phases, compositions, outputs, ref_phase in cases:
        values = {out: [] for out in outputs}
        for composition in compositions:
            eq = _equilibrium(dbf, elements, phases, T, composition, output=['HM', 'SM', 'CPM'])
            stable = sorted(phase for phase, _, _ in _stable(eq, dbf, elements))
            all_phases = sorted(phase for phase, _, _ in _stable(_equilibrium(dbf, elements, list(dbf.phases), T, composition), dbf, elements))
            if stable != sorted(phases) or all_phases != sorted(phases):
                raise RuntimeError(f'Expected {sorted(phases)} at T={T}, X={composition}, got {stable} ({all_phases} with all phases)')
            x = dict(zip(elements, [1 - sum(composition)] + list(composition)))
            for out in outputs:
                base = out[:-1] if out.endswith('R') else out
                value = float(getattr(eq, base).values.squeeze())
                if out.endswith('R'):
                    value -= sum(x[el] * _pure_property(dbf, el, ref_phase, base, T) for el in elements)
                values[out].append(_round(value))
        for out in outputs:
            dataset = {
                'components': elements + ['VA'], 'phases': phases, 'output': out,
                'conditions': {'P': P, 'T': [T], **{f'X_{el}': [c[i] for c in compositions] for i, el in enumerate(elements[1:])}},
                'values': [[values[out]]],
                'reference': REFERENCE, 'comment': f'{out} in {" + ".join(phases)}',
            }
            if out.endswith('R'):
                dataset['reference_states'] = {el: {'phase': ref_phase} for el in elements}
            datasets[f'eq_{name}_{out}'] = dataset
    return datasets


def _site_fractions(mod, configuration, occupancies):
    """Site fractions of a configuration, zero for constituents it does not contain."""
    values = {y: 0.0 for y in mod.site_fractions}
    for subl_idx, (subl, occ) in enumerate(zip(configuration, occupancies)):
        subl, occ = (subl, occ) if isinstance(subl, list) else ([subl], [occ])
        for species, fraction in zip(subl, occ):
            values[v.Y(mod.phase_name, subl_idx, species)] = float(fraction)
    return values


def fixed_configuration_value(dbf, elements, phase, prop, configuration, occupancies, T, exclusions=()):
    mod = Model(dbf, elements + ['VA'], phase)
    for contrib in exclusions:
        mod.models[contrib] = 0
        mod.endmember_reference_model.models[contrib] = 0
    point = {v.T: T, v.P: P, **_site_fractions(mod, configuration, occupancies)}
    if prop.endswith('_MIX'):
        return float(getattr(mod, prop).xreplace(point))
    base = prop.split('_')[0]
    value = float(getattr(mod, base).xreplace(point))
    for el in elements:
        value -= float(mod.moles(el).xreplace(point)) * _pure_property(dbf, el, 'FCC_A1', base, T)
    return value


def fixed_configuration_dataset(dbf, elements, phase, prop, site_ratios, configurations, occupancies, exclusions=()):
    values = [[[_round(fixed_configuration_value(dbf, elements, phase, prop, config, occ, T, exclusions))
                for config, occ in zip(configurations, occupancies)] for T in FIXED_TEMPERATURES]]
    dataset = {
        'components': elements + ['VA'], 'phases': [phase], 'output': prop,
        'solver': {'mode': 'manual', 'sublattice_site_ratios': site_ratios,
                   'sublattice_configurations': configurations, 'sublattice_occupancies': occupancies},
        'conditions': {'P': P, 'T': FIXED_TEMPERATURES},
        'values': values,
        'reference': REFERENCE, 'comment': f'{prop} of {phase}',
    }
    if exclusions:
        dataset['excluded_model_contributions'] = list(exclusions)
    return dataset


def fixed_configuration_datasets(dbf, system):
    elements = SYSTEMS[system]
    datasets = {}
    props = {'TAU': ['HM_FORM', 'SM_FORM', 'CPM_FORM', 'HM_MIX', 'SM_MIX', 'CPM_MIX'], 'ORD': ['HM_MIX', 'HM_FORM'],
             'LIQUID': ['HM_MIX', 'SM_MIX', 'CPM_MIX']}
    for phase, (site_ratios, configurations, occupancies) in FIXED_CONFIGURATIONS[system].items():
        for prop in props[phase]:
            datasets[f'fixed_{phase.lower()}_{prop}'] = fixed_configuration_dataset(dbf, elements, phase, prop, site_ratios, configurations, occupancies)
    site_ratios, configurations, occupancies = FIXED_CONFIGURATIONS[system]['LIQUID']
    # ESPEI applies excluded_model_contributions to _FORM data's reference states only, so the exclusion is on _MIX data
    datasets['fixed_liquid_SM_MIX_no_idmix'] = fixed_configuration_dataset(dbf, elements, 'LIQUID', 'SM_MIX', site_ratios, configurations,
                                                                           occupancies, exclusions=['idmix'])
    return datasets


def generate(system):
    dbf = Database(os.path.join(HERE, 'abcd.tdb'))
    return {**zpf_datasets(dbf, system), **activity_datasets(dbf, system),
            **equilibrium_property_datasets(dbf, system), **fixed_configuration_datasets(dbf, system)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument('--check', action='store_true', help='compare with the committed JSON instead of writing it')
    args = parser.parse_args()
    warnings.filterwarnings('ignore')
    mismatched = []
    for system in SYSTEMS:
        path = os.path.join(HERE, f'{system}.json')
        datasets = json.loads(json.dumps(generate(system)))
        if args.check:
            with open(path) as fp:
                if json.load(fp) != datasets:
                    mismatched.append(path)
        else:
            with open(path, 'w') as fp:
                json.dump(datasets, fp, indent=1)
                fp.write('\n')
    if mismatched:
        print('Regenerated datasets differ from', ', '.join(mismatched))
        sys.exit(1)


if __name__ == '__main__':
    main()
