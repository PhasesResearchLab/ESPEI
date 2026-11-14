"""
Reference states of fixed configuration property data.

See the Reference States page of the documentation for the definitions.
"""

import copy
import itertools
import warnings
from typing import Mapping, Optional, Sequence, Tuple, Type

import symengine
from symengine import S, Symbol
from pycalphad import Database, Model, variables as v

from espei.database_utils import _param_present_in_database

# pycalphad's Model derives these properties from the Gibbs energy of its
# contributions (Model.models). Their reference states are built from the
# Gibbs energy, which is finite for endmembers without atoms (e.g. VA:VA).
_GIBBS_ENERGY_PROPERTIES = {"GM", "HM", "SM", "CPM"}


def _split_property_reference(output: str) -> Tuple[str, str]:
    """Split a dataset output into the property and the reference state, e.g. ``"HM_FORM"`` -> ``("HM", "_FORM")``."""
    for reference in ("_FORM", "_MIX"):
        if output.endswith(reference):
            return output[:-len(reference)], reference
    return output, ""


def _endmember_point(model: Model, endmember: Sequence[v.Species]):
    return {v.SiteFraction(model.phase_name, idx, sp): float(sp == endmember[idx])
            for idx, sublattice in enumerate(model.constituents) for sp in sublattice}


def _ser_value(dbf: Database, element: str, per_formula_unit, gibbs_energy: bool, model_classes: Mapping[str, Type[Model]], parameters) -> symengine.Expr:
    """Value per mole of atoms of the pure element in its SER phase."""
    phase_name = dbf.refstates.get(element, {}).get("phase")
    if phase_name not in dbf.phases:
        ghser = "GHSER" + (element.upper()*2)[:2]
        if not gibbs_energy:
            raise ValueError(f"The SER phase of {element}, {phase_name}, is not in the Database.")
        warnings.warn(f"The SER phase of {element}, {phase_name}, is not in the Database. The {ghser} function is used as the "
                      f"reference of {element} in _FORM data instead, without physical contributions such as magnetism.")
        symbols = {Symbol(s): val for s, val in dbf.symbols.items() if parameters is None or s not in parameters}
        return Model.symbol_replace(Symbol(ghser), symbols)
    model = model_classes.get(phase_name, Model)(dbf, [element, "VA"], phase_name, parameters=parameters)
    endmembers = [em for em in itertools.product(*[sorted(subl) for subl in model.constituents])
                  if sum(sp.number_of_atoms for sp in em) > 0]
    if len(endmembers) != 1:
        raise NotImplementedError(f"The SER phase of {element}, {phase_name}, has {len(endmembers)} configurations of pure {element}. Only one is supported.")
    if not _param_present_in_database(dbf, phase_name, [sp.name for sp in endmembers[0]], "G"):
        raise ValueError(f"The SER phase of {element}, {phase_name}, has no Gibbs energy parameter for pure {element}.")
    point = _endmember_point(model, endmembers[0])
    return per_formula_unit(model).xreplace(point) / model._site_ratio_normalization.xreplace(point)


def _reference_property(model: Model, prop: str, reference: str, excluded_contributions: Sequence[str] = (),
                        dbf: Optional[Database] = None, model_classes: Optional[Mapping[str, Type[Model]]] = None,
                        parameters=None) -> symengine.Expr:
    """
    Return a property of the reference state of a phase, per mole of atoms of the phase.

    Parameters
    ----------
    model : Model
        Model of the phase.
    prop : str
        Model property, e.g. ``"HM"`` or ``"VM"``.
    reference : str
        ``""`` (absolute, no reference), ``"_MIX"`` or ``"_FORM"``.
    excluded_contributions : Sequence[str]
        Model contributions to remove from the reference, e.g. ``["mag"]``.
    dbf : Optional[Database]
        Database with the SER phases. Only used for ``_FORM``.
    model_classes : Optional[Mapping[str, Type[Model]]]
        Model classes of the SER phases. Defaults to ``Model``.
    parameters : Optional[List[str]]
        Parameters to keep symbolic in the SER phases, as in ``Model``.

    Returns
    -------
    symengine.Expr

    """
    if reference == "":
        return S.Zero
    gibbs_energy = prop in _GIBBS_ENERGY_PROPERTIES

    def per_formula_unit(mod):
        if gibbs_energy:
            return symengine.Add(*[val for name, val in mod.models.items() if name not in excluded_contributions])
        return getattr(mod, prop) * mod._site_ratio_normalization

    if reference == "_MIX":
        # endmembers weighted by the product of their site fractions
        quantity = per_formula_unit(model)
        value = symengine.Add(*[symengine.Mul(*[v.SiteFraction(model.phase_name, idx, sp) for idx, sp in enumerate(endmember)])
                                * quantity.xreplace(_endmember_point(model, endmember))
                                for endmember in itertools.product(*[sorted(subl) for subl in model.constituents])])
    elif reference == "_FORM":
        # pure elements in their SER phases weighted by the moles of each element
        value = symengine.Add(*[model.moles(element, per_formula_unit=True) * _ser_value(dbf, element, per_formula_unit, gibbs_energy, model_classes or {}, parameters)
                                for element in model.nonvacant_elements if element != "/-"])
    else:
        raise ValueError(f'Unknown reference state "{reference}". Expected "", "_FORM" or "_MIX".')
    if gibbs_energy:
        # pycalphad derives the property from the reference Gibbs energy, like for the phase
        reference_model = copy.copy(model)
        reference_model.models = {"reference": value}
        return getattr(reference_model, prop)
    return value / model._site_ratio_normalization


def _set_referenced_property(model: Model, output: str, excluded_contributions: Sequence[str] = (), dbf: Optional[Database] = None,
                             model_classes: Optional[Mapping[str, Type[Model]]] = None, parameters=None) -> str:
    """Set the property of a dataset output relative to its reference state on the model, e.g. ``HMR`` for ``HM_FORM``, and return its name."""
    prop, reference = _split_property_reference(output)
    if reference == "":
        return output
    name = prop + "R"
    reference_property = _reference_property(model, prop, reference, excluded_contributions, dbf, model_classes, parameters)
    setattr(model, name, getattr(model, prop) - reference_property)
    return name
