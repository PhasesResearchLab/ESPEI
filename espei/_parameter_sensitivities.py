"""
Parameter sensitivities of converged equilibria, for likelihood gradients, computed in NumPy.

Signatures and return layouts follow pycalphad-core's ``Workspace.get_tangent`` and
``Workspace.get_gradient``. Each point's solver state is rebuilt from the Workspace's results, and
the per-phase arrays are read from the ``__getstate__`` tuples of PyCalphad's solver classes.
"""
import weakref
from collections import OrderedDict
from itertools import repeat
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from symengine import lambdify, oo, zoo
from pycalphad import variables as v
from pycalphad.codegen.sympydiff_utils import _get_lambidfy_options
from pycalphad.core.cache import cacheit
from pycalphad.core.minimizer import construct_equilibrium_system
from pycalphad.core.solver import Solver
from pycalphad.property_framework import as_property
from pycalphad.property_framework.computed_property import ModelComputedProperty

BACKEND = 'numpy'

# Positions in the __getstate__ tuples of pycalphad/core/minimizer.pyx
_COMPSET_STATE_LENGTH = 16
_E_MATRIX, _MASS_JAC, _C_STATEVARS, _C_COMPONENT, _MOLES_NORMALIZATION, _MOLES_NORMALIZATION_GRAD = 5, 7, 9, 10, 11, 13
_SYSTEM_STATE_LENGTH = 19
_PHASE_AMT, _FREE_STABLE_COMPSETS, _SYSTEM_AMOUNT, _MOLE_FRACTIONS = 7, 16, 17, 18
_SYSTEM_SPEC_LENGTH = 11
_MOLE_FRACTION_COEFFICIENTS, _FREE_CHEMICAL_POTENTIALS, _FREE_STATEVARS, _FIXED_CHEMICAL_POTENTIALS, _FIXED_COMPSETS = 4, 6, 7, 8, 10


@cacheit
def _parameter_kernels(expr, variables, parameters, include_mixed):
    """Compiled dZ/dtheta and, with include_mixed, d2Z/dy dtheta (site fractions only) for every parameter."""
    opts = _get_lambidfy_options(None)
    inp = tuple(variables) + tuple(parameters)
    grad = [expr.diff(p).xreplace({zoo: oo}) for p in parameters]
    mixed = None
    if include_mixed:
        site_fractions = [x for x in variables if isinstance(x, v.SiteFraction)]
        mixed = lambdify(inp, [[g.diff(y).xreplace({zoo: oo}) for y in site_fractions] for g in grad], **opts)
    return lambdify(inp, grad, **opts), mixed


_FACTORY_KERNELS = weakref.WeakKeyDictionary()


def _phase_kernels(phase_record_factory, phase_name, property_name, include_mixed):
    kernels = _FACTORY_KERNELS.setdefault(phase_record_factory, {})
    key = (phase_name, property_name, include_mixed)
    if key not in kernels:
        mod = phase_record_factory.models[phase_name]
        expr = getattr(mod, property_name)
        # undefined symbols are zero, as in PhaseRecordFactory.get_phase_property
        undefs = {x for x in expr.free_symbols if not isinstance(x, v.StateVariable)} - set(phase_record_factory.param_symbols)
        expr = expr.xreplace(dict(zip(undefs, repeat(0.))))
        variables = tuple(phase_record_factory.state_variables + mod.site_fractions)
        kernels[key] = _parameter_kernels(expr, variables, tuple(phase_record_factory.param_symbols), include_mixed)
    return kernels[key]


def _parameter_indices(phase_record, inputs: Sequence[str]) -> np.ndarray:
    names = [str(s) for s in phase_record.phase_record_factory.param_symbols]
    missing = [str(name) for name in inputs if str(name) not in names]
    if missing:
        raise ValueError(f"{missing} are not parameters of the phase records ({names})")
    return np.array([names.index(str(name)) for name in inputs], dtype=int)


def _parameter_partials(compset, property_name, mixed=False):
    """dZ/dtheta (and d2Z/dy dtheta) of a phase's model property at its degrees of freedom, for every parameter."""
    phase_record = compset.phase_record
    num_params = len(phase_record.phase_record_factory.param_symbols)
    if num_params == 0:
        return (np.zeros(0), np.zeros((0, phase_record.phase_dof))) if mixed else np.zeros(0)
    grad, grad_mixed = _phase_kernels(phase_record.phase_record_factory, phase_record.phase_name, property_name, mixed)
    inp = np.concatenate([np.asarray(compset.dof, dtype=np.float64), np.asarray(phase_record.parameters, dtype=np.float64)])
    partials = np.reshape(grad(inp), (num_params,))
    if mixed:
        return partials, np.reshape(grad_mixed(inp), (num_params, phase_record.phase_dof))
    return partials


def _moles_per_formula_unit(compset) -> float:
    moles, total = np.zeros(1), 0.0
    for el_idx in range(len(compset.phase_record.nonvacant_elements)):
        moles[:] = 0
        compset.phase_record.formulamole_obj(moles, compset.dof, el_idx)
        total += moles[0]
    return total


def _points(wks):
    """Per point: None without a converged equilibrium, else (composition sets, conditions, chemical potentials)."""
    str_conds_keys = [str(k) for k in wks.eq.coords.keys() if k not in ('vertex', 'component', 'internal_dof')]
    conds_keys = [None] * len(str_conds_keys)
    for k in wks.conditions.keys():
        conds_keys[str_conds_keys.index(str(k))] = k
    for index, compsets in wks.enumerate_composition_sets():
        mu = np.asarray(wks.eq.MU[index], dtype=np.float64)
        if len(compsets) == 0 or np.any(np.isnan(mu)):
            yield None
            continue
        cur_conds = OrderedDict(zip(conds_keys, [float(wks.eq.coords[b][a]) for a, b in zip(index, str_conds_keys)]))
        yield compsets, cur_conds, mu


def _solver_state(compsets, cur_conds, chemical_potentials):
    """SystemSpecification and recomputed SystemState of a converged point, and their checked __getstate__ tuples."""
    spec = Solver().get_system_spec(compsets, cur_conds)
    state = spec.get_new_state(compsets)
    state.chemical_potentials[:] = chemical_potentials
    state.recompute(spec)
    spec_tuple, state_tuple = spec.__getstate__(), state.__getstate__()
    if len(spec_tuple) != _SYSTEM_SPEC_LENGTH or len(state_tuple) != _SYSTEM_STATE_LENGTH:
        raise RuntimeError('Unexpected SystemSpecification or SystemState layout')
    if len(spec_tuple[_FREE_STATEVARS]) > 0 or len(spec_tuple[_FIXED_CHEMICAL_POTENTIALS]) > 0 or len(spec_tuple[_FIXED_COMPSETS]) > 0:
        raise NotImplementedError('Parameter sensitivities need fixed state variables and no chemical potential or fixed phase conditions')
    return spec, state, spec_tuple, state_tuple


def _tangent(compsets, cur_conds, chemical_potentials, inputs):
    """Tangent of one point with respect to the named parameters."""
    phase_record = compsets[0].phase_record
    indices = _parameter_indices(phase_record, inputs)
    num_inputs, num_statevars = len(indices), len(phase_record.state_variables)
    num_components = len(phase_record.nonvacant_elements)
    spec, state, spec_tuple, state_tuple = _solver_state(compsets, cur_conds, chemical_potentials)
    coefficients = np.asarray(spec_tuple[_MOLE_FRACTION_COEFFICIENTS]).reshape(-1, num_components)
    free_mu = np.asarray(spec_tuple[_FREE_CHEMICAL_POTENTIALS], dtype=int)
    phase_amt = np.asarray(state_tuple[_PHASE_AMT])
    free_stable = [int(i) for i in state_tuple[_FREE_STABLE_COMPSETS]]
    system_amount, mole_fractions = state_tuple[_SYSTEM_AMOUNT], np.asarray(state_tuple[_MOLE_FRACTIONS])
    num_phases, num_molefracs = len(free_stable), coefficients.shape[0]
    J, _ = construct_equilibrium_system(spec, state, 0)
    # The parameters' columns of the equilibrium matrix, as if they were free state variables
    columns = np.zeros((J.shape[0], num_inputs))
    c_parameters, compset_arrays = [], []
    for idx, compset in enumerate(compsets):
        cs_tuple = state.cs_states[idx].__getstate__()
        num_dof = compset.phase_record.phase_dof
        E, mass_jac = np.asarray(cs_tuple[_E_MATRIX]), np.asarray(cs_tuple[_MASS_JAC])
        c_statevars, c_component = np.asarray(cs_tuple[_C_STATEVARS]), np.asarray(cs_tuple[_C_COMPONENT])
        moles_normalization_grad = np.asarray(cs_tuple[_MOLES_NORMALIZATION_GRAD])
        if (len(cs_tuple) != _COMPSET_STATE_LENGTH or E.shape[0] < num_dof or mass_jac.shape != (num_components, num_statevars + num_dof)
                or c_component.shape != (num_components, num_dof) or c_statevars.shape != (num_dof, num_statevars)):
            raise RuntimeError('Unexpected CompsetState layout')
        dG, d2G = _parameter_partials(compset, 'G', mixed=True)
        c_theta = -E[:num_dof, :num_dof] @ d2G[indices].T
        c_parameters.append(c_theta)
        compset_arrays.append((c_statevars, c_component, cs_tuple[_MOLES_NORMALIZATION], moles_normalization_grad[num_statevars:]))
        if idx not in free_stable:
            continue
        columns[free_stable.index(idx)] = -dG[indices]
        dM = mass_jac[:, num_statevars:]
        columns[num_phases:num_phases + num_molefracs] += (phase_amt[idx] / system_amount) * \
            (coefficients @ (dM - np.outer(mole_fractions, moles_normalization_grad[num_statevars:]))) @ c_theta
        columns[num_phases + num_molefracs] += phase_amt[idx] * (moles_normalization_grad[num_statevars:] @ c_theta)
    rhs = -columns
    if not (np.all(np.isfinite(J)) and np.all(np.isfinite(rhs))):
        solution = np.full((J.shape[1], num_inputs), np.nan)
    else:
        solution = np.linalg.lstsq(J, rhs, rcond=1e-16)[0]
        residuals = np.linalg.norm(J @ solution - rhs, axis=0)
        # infeasible unless the residual is small relative to the right-hand side or at rounding level
        rounding = 1e3 * np.finfo(np.float64).eps * np.linalg.norm(J) * np.linalg.norm(solution, axis=0)
        solution[:, (residuals > 1e-3 * np.linalg.norm(rhs, axis=0)) & (residuals > rounding)] = np.nan
    delta_mu = np.zeros((num_inputs, num_components))
    delta_mu[:, free_mu] = solution[:len(free_mu)].T
    delta_statevars = np.zeros((num_inputs, num_statevars))
    delta_formula_units = np.zeros((num_inputs, len(compsets)))
    delta_formula_units[:, free_stable] = solution[len(free_mu):len(free_mu) + num_phases].T
    delta_phase_amounts = np.zeros((num_inputs, len(compsets)))
    delta_sitefracs = []
    for idx, (c_statevars, c_component, moles_normalization, dM_dy) in enumerate(compset_arrays):
        delta_y = delta_mu @ c_component + delta_statevars @ c_statevars.T + c_parameters[idx].T
        delta_sitefracs.append(delta_y)
        # NP = phase_amt * M, in moles of atoms
        delta_phase_amounts[:, idx] = delta_formula_units[:, idx] * moles_normalization + phase_amt[idx] * (delta_y @ dM_dy)
    return {'chemical_potentials': delta_mu, 'state_variables': delta_statevars,
            'phase_amounts': delta_phase_amounts, 'site_fractions': delta_sitefracs}


def get_tangent(wks, inputs: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    """Tangent sensitivities of every point's equilibrium state with respect to the named parameters."""
    return [None if point is None else _tangent(*point, inputs) for point in _points(wks)]


def _property_gradient(prop, compsets, tangent, inputs):
    """Contract one point's tangent with a property's partial derivatives."""
    indices = _parameter_indices(compsets[0].phase_record, inputs)
    if isinstance(prop, v.ChemicalPotential):
        elements = list(compsets[0].phase_record.nonvacant_elements)
        if prop.species.name not in elements:
            raise NotImplementedError(f'Gradients of {prop} need a pure element')
        return tangent['chemical_potentials'][:, elements.index(prop.species.name)]
    phase_name, multiplicity = None, 1
    if prop.phase_name is not None:
        tokens = prop.phase_name.split('#')
        phase_name, multiplicity = tokens[0], (int(tokens[1]) if len(tokens) > 1 else 1)
    name = prop.model_attr_name.encode('utf-8')
    num_statevars = len(compsets[0].phase_record.state_variables)
    gradient, seen = None, 0
    for idx, compset in enumerate(compsets):
        if phase_name is not None:
            if compset.phase_record.phase_name != phase_name:
                continue
            seen += 1
            if seen != multiplicity:
                continue
        elif compset.NP == 0:
            continue
        value, grad = np.zeros(1), np.zeros(compset.dof.shape[0])
        compset.phase_record.prop(value, compset.dof, name)
        if np.isnan(value[0]):
            continue
        compset.phase_record.prop_grad(grad, compset.dof, name)
        partial = tangent['state_variables'] @ grad[:num_statevars] + tangent['site_fractions'][idx] @ grad[num_statevars:] \
            + _parameter_partials(compset, prop.model_attr_name)[indices]
        if phase_name is not None:
            return partial
        gradient = (0.0 if gradient is None else gradient) + tangent['phase_amounts'][:, idx] * value[0] + compset.NP * partial
    return np.full(len(inputs), np.nan) if gradient is None else gradient


def get_gradient(wks, prop, inputs: Sequence[str]) -> np.ndarray:
    """Derivatives of a property with respect to the named parameters at every point."""
    prop = as_property(prop)
    if not isinstance(prop, (ModelComputedProperty, v.ChemicalPotential)):
        raise NotImplementedError(f'Gradients of {prop} are not implemented')
    gradients = []
    for point in _points(wks):
        if point is None:
            gradients.append(np.full(len(inputs), np.nan))
            continue
        compsets, cur_conds, mu = point
        gradients.append(_property_gradient(prop, compsets, _tangent(compsets, cur_conds, mu, inputs), inputs))
    conds_shape = tuple(len(wks.eq.coords[str(b)]) for b in wks.condition_axis_order)
    return np.reshape(np.asarray(gradients, dtype=np.float64), conds_shape + (len(inputs),))


def isolated_phase_gm_gradient(compset, inputs: Sequence[str]) -> np.ndarray:
    """dGM/dtheta of a converged single-phase constrained minimum, by the envelope theorem."""
    indices = _parameter_indices(compset.phase_record, inputs)
    return _parameter_partials(compset, 'G')[indices] / _moles_per_formula_unit(compset)


def fixed_configuration_gradient(compset, prop: str, inputs: Sequence[str]) -> np.ndarray:
    """Partial derivatives of a model property at the CompositionSet's fixed state variables and site fractions."""
    indices = _parameter_indices(compset.phase_record, inputs)
    return _parameter_partials(compset, prop)[indices]
