
from espei.optimizers.opt_base import OptimizerBase
from espei.optimizers.graph import OptNode
from pycalphad import Database
from scipy.optimize import minimize
import numpy as np

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
        return OptNode(dict(zip(symbol_names, result.x)), datasets)

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

def test_optimizer_can_commit():
    """Test that TestOptimizer can call commit after a fit"""
    syms = ['A', 'B', 'C']
    targ = [-100, 0, 10]
    opt = TestOptimizer(Database())
    opt.fit(syms, {}, targ)
    opt.commit()
    assert len(opt.graph.nodes) == 2  # root and new commit


def test_optimizer_can_commit_multiple():
    """Test that TestOptimizer can commit multiple fit nodes to the graph"""
    syms = ['A', 'B', 'C']
    targ = [-100, 0, 10]
    opt = TestOptimizer(Database())
    opt.fit(syms, {}, targ)
    opt.commit()
    opt.fit(syms, {}, targ)
    opt.commit()
    assert len(opt.graph.nodes) == 3  # root and two new commits


def test_optimizer_can_discard_changes():
    """Test that TestOptimizer can discard commits and reset back to the previous state"""
    syms = ['A', 'B', 'C']
    targ = [-100, 0, 10]
    opt = TestOptimizer(Database())
    opt.fit(syms, {}, targ)
    opt.commit()
    # new head parameters should be -100, 0, 0
    opt.fit(syms, {}, [1000, 1000, 1000])
    opt.fit(syms, {}, [123, 456, 789])
    opt.discard()
    for sym, t in zip(syms, targ):
        assert sym in opt.dbf.symbols
        assert np.isclose(opt.dbf.symbols[sym], t)

