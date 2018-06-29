"""
Uses matplotlib functionality to test graphical outputs
"""

from matplotlib.testing.decorators import image_comparison
from espei.plot import dataplot
from pycalphad import variables as v
from espei.tests.testing_data import A_B_DATASET_PHASE_EQUILIBRIA
from espei.utils import PickleableTinyDB, MemoryStorage

@image_comparison(baseline_images=['dataplot_binary_equilibria_types'], extensions=['png'])
def test_dataplot_plots_equilibria_types():
    """Dataplot should be able to reproduce a single boundary (null) equilibria, a tieline, and a 3 phase equilibria"""
    ds = PickleableTinyDB(storage=MemoryStorage)
    ds.insert(A_B_DATASET_PHASE_EQUILIBRIA)

    comps = ['A', 'B']
    phases = ["PHASE_1", "PHASE_2", "PHASE_3"]
    conds = {v.P: 101325, v.T: (0, 400, 40), v.X('B'): (0, 1, 0.01)}

    ax = dataplot(comps, phases, conds, ds)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 400)

