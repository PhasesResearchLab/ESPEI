"""
Parameter sensitivities of converged equilibria, for likelihood gradients.

Signatures and return layouts follow pycalphad-core's ``Workspace.get_tangent`` and
``Workspace.get_gradient``. This module is a stub: install a backend to compute gradients.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

BACKEND = None


def get_tangent(wks, inputs: Sequence[str]) -> List[Optional[Dict[str, Any]]]:
    """Tangent sensitivities of every point's equilibrium state with respect to the named parameters."""
    raise NotImplementedError("no parameter-sensitivity backend")


def get_gradient(wks, prop, inputs: Sequence[str]) -> np.ndarray:
    """Derivatives of a property with respect to the named parameters at every point."""
    raise NotImplementedError("no parameter-sensitivity backend")


def isolated_phase_gm_gradient(compset, inputs: Sequence[str]) -> np.ndarray:
    """dGM/dtheta of a converged single-phase constrained minimum, by the envelope theorem."""
    raise NotImplementedError("no parameter-sensitivity backend")


def fixed_configuration_gradient(compset, prop: str, inputs: Sequence[str]) -> np.ndarray:
    """Partial derivatives of a model property at the CompositionSet's fixed state variables and site fractions."""
    raise NotImplementedError("no parameter-sensitivity backend")
