
from espei.error_functions.context import setup_context
from espei.optimizers.opt_base import OptimizerBase
from espei.optimizers.opt_scipy import SciPyOptimizer
from pycalphad import Database
from scipy.optimize import minimize
import numpy as np
import pytest

from .fixtures import datasets_db
from .testing_data import A_B_DATASET_HM_MIX_ALPHA, A_B_REGULAR_SOLUTION_TDB

class TestOptimizer(OptimizerBase):
    def _fit(self, symbol_names, datasets, target_values=None, initial_guess=None):
        symbol_names = np.array(symbol_names)
        target_values = np.array(target_values)
        if target_values is None:
            target_values = np.zeros_like(symbol_names)
        if initial_guess is None:
            initial_guess = np.random.random(target_values.shape)
        ctx = {'target': target_values}
        result = minimize(self.predict, initial_guess, method='Powell', args=(ctx,))
        return dict(zip(symbol_names, result.x))

    @staticmethod
    def predict(params, context):
        target = context['target']
        return np.sum(np.square(target - params))

def test_optimizer_can_fit():
    """Test that TestOptimizer can call fit with the proper API"""
    syms = ['A', 'B', 'C']
    targ = [-100, 0, 10]
    opt = TestOptimizer(Database())
    opt.fit(syms, {}, targ)
    for sym, t in zip(syms, targ):
        assert sym in opt.dbf.symbols
        assert np.isclose(opt.dbf.symbols[sym], t)


@pytest.mark.parametrize('kwargs', [{}, {'method': 'L-BFGS-B', 'jac': True}], ids=['Powell', 'L-BFGS-B'])
def test_scipy_optimizer_maximizes_the_likelihood(datasets_db, kwargs):
    """SciPyOptimizer fits L = 4 HM_MIX(x_B = 0.5), and neither it nor setup_context modifies the Database"""
    datasets_db.insert({**A_B_DATASET_HM_MIX_ALPHA, 'values': [[[500.0]]]})
    dbf = Database(A_B_REGULAR_SOLUTION_TDB)
    symbols_before = dict(dbf.symbols)
    setup_context(dbf, datasets_db, ['L_ALPHA'])
    assert dbf.symbols == symbols_before
    fit_dbf = SciPyOptimizer(dbf).fit(['L_ALPHA'], datasets_db, **kwargs)
    assert fit_dbf.symbols['L_ALPHA'] == pytest.approx(2000.0, rel=1e-4)
    assert dbf.symbols == symbols_before
