"""
Tests for plotting functions.

Mostly integration tests that don't validate the plots themselves, but rather that the
internal usage and matplotlib usage is correct.
"""

import matplotlib.pyplot as plt
from pycalphad import Database, variables as v

from espei.plot import dataplot, plot_endmember, plot_interaction

from .fixtures import datasets_db
from .testing_data import *

def test_dataplot_runs(datasets_db):
    """Test that dataplot runs without an error."""

    datasets_db.insert(CU_MG_DATASET_ZPF_ZERO_ERROR)  # Full tie-line
    datasets_db.insert(CU_MG_DATASET_ZPF_WORKING)  # Half tie-line

    comps = ['CU', 'MG', 'VA']
    phases = ['CUMG2', 'FCC_A1', 'HCP_A3', 'LAVES_C15', 'LIQUID']
    conds = {v.P: 101325, v.T: (300, 2000, 10), v.X('MG'): (0, 1, 0.01)}
    fig = plt.figure()
    dataplot(comps, phases, conds, datasets_db)
    # fig.savefig('test_dataplot_runs-figure.png')
    plt.close(fig)


def test_dataplot_runs_ternary_isothermal(datasets_db):
    """Test that dataplot runs without an error for a ternary isothermal case."""

    datasets_db.insert(A_B_C_DATASET_TERNARY_PHASE_EQUILIBRIA)

    comps = ['A', 'B', 'C', 'VA']
    phases = ['PHASE_1', 'PHASE_2']
    conds = {v.P: 101325, v.T: 300.0, v.X('B'): (0, 1, 0.01), v.X('C'): (0, 1, 0.01)}
    fig = plt.figure()
    dataplot(comps, phases, conds, datasets_db)
    # fig.savefig('test_dataplot_runs-figure.png')
    plt.close(fig)


def test_comprehensive_binary_dataplot_test(datasets_db):
    """Test that dataplot runs for a large variety of cases."""
    datasets_db.insert_multiple([
        {"reference": "One-phase", "comment": "", "components": ["A", "B"], "phases": ["ALPHA"], "conditions": {"P": 101325,"T": 300}, "output": "ZPF", "values": [[["ALPHA", ["B"], [0.5]]]]},
        {"reference": "One-phase w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA"], "conditions": {"P": 101325,"T": 400}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [0.5]]]]},
        {"reference": "One-phase (null) w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA"], "conditions": {"P": 101325,"T": 500}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [None]]]]},
        {"reference": "Two-phase", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 600}, "output": "ZPF", "values": [[["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.75]]]]},
        {"reference": "Two-phase (1 null)", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 700}, "output": "ZPF", "values": [[["ALPHA", ["B"], [None]], ["BETA", ["B"], [0.75]]]]},
        {"reference": "Two-phase w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 800}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.75]]]]},
        {"reference": "Two-phase (1 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 900}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [None]], ["BETA", ["B"], [0.75]]]]},
        {"reference": "Two-phase (2 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [None]], ["BETA", ["B"], [None]]]]},
        {"reference": "Three-phase", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1100}, "output": "ZPF", "values": [[["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.625]], ["GAMMA", ["B"], [0.75]]]]},
        {"reference": "Three-phase (1 null)", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1100}, "output": "ZPF", "values": [[["ALPHA", ["B"], [None]], ["BETA", ["B"], [0.625]], ["GAMMA", ["B"], [0.75]]]]},
        {"reference": "Three-phase (2 null)", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1200}, "output": "ZPF", "values": [[["ALPHA", ["B"], [None]], ["BETA", ["B"], [None]], ["GAMMA", ["B"], [0.75]]]]},
        {"reference": "Three-phase w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1300}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [0.25]], ["BETA", ["B"], [0.625]], ["GAMMA", ["B"], [0.75]]]]},
        {"reference": "Three-phase (1 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1400}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [None]], ["BETA", ["B"], [0.625]], ["GAMMA", ["B"], [0.75]]]]},
        {"reference": "Three-phase (2 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1500}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [None]], ["BETA", ["B"], [None]], ["GAMMA", ["B"], [0.75]]]]},
        {"reference": "Three-phase (3 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1600}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B"], [0.5]], ["ALPHA", ["B"], [None]], ["BETA", ["B"], [None]], ["GAMMA", ["B"], [None]]]]},
    ])

    comps = ["A", "B"]
    phases = ["ALPHA", "BETA", "GAMMA"]
    conds = {v.P: 101325, v.T: (200, 1700, 10), v.X('B'): (0, 1, 0.01)}
    fig, ax = plt.subplots()
    dataplot(comps, phases, conds, datasets_db, ax=ax)
    # fig.savefig('test_comprehensive_binary_dataplot_test.png', bbox_inches="tight")
    plt.close(fig)


def test_comprehensive_ternary_dataplot_test(datasets_db):
    """Test that dataplot runs for a large variety of cases."""
    datasets_db.insert_multiple([
        {"reference": "One-phase", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["ALPHA", ["B", "C"], [0.05, 0.05]]]]},
        {"reference": "One-phase w/ HYPERPLANE", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.10, 0.05]], ["ALPHA", ["B", "C"], [0.10, 0.05]]]]},
        {"reference": "One-phase (null) w/ HYPERPLANE", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.05, 0.10]], ["ALPHA", ["B", "C"], [None, None]]]]},

        {"reference": "Two-phase", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["ALPHA", ["B", "C"], [0.30, 0.05]], ["BETA", ["B", "C"], [0.60, 0.05]]]]},
        {"reference": "Two-phase (1 null)", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [0.40, 0.10]]]]},
        {"reference": "Two-phase w/ HYPERPLANE", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.4, 0.15]], ["ALPHA", ["B", "C"], [0.10, 0.15]], ["BETA", ["B", "C"], [0.70, 0.15]]]]},
        {"reference": "Two-phase (1 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.35, 0.20]], ["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [0.5, 0.20]]]]},
        {"reference": "Two-phase (2 null) w/ HYPERPLANE", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.3, 0.25]], ["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [None, None]]]]},

        {"reference": "Three-phase", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["ALPHA", ["B", "C"], [0.2, 0.40]], ["BETA", ["B", "C"], [0.5, 0.40]], ["GAMMA", ["B", "C"], [0.275, 0.45]]]]},
        {"reference": "Three-phase (1 null)", "comment": "Doesn't plot", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [0.1, 0.45]], ["GAMMA", ["B", "C"], [0.3, 0.45]]]]},
        {"reference": "Three-phase (2 null)", "comment": "Doesn't plot", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [None, None]], ["GAMMA", ["B", "C"], [0.40, 0.50]]]]},
        {"reference": "Three-phase w/ HYPERPLANE", "comment": "", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.10, 0.775]], ["ALPHA", ["B", "C"], [0.05, 0.75]], ["BETA", ["B", "C"], [0.15, 0.75]], ["GAMMA", ["B", "C"], [0.10, 0.80]]]]},
        {"reference": "Three-phase (1 null) w/ HYPERPLANE", "comment": "Doesn't plot", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.5, 0.80]], ["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [0.10, 0.80]], ["GAMMA", ["B", "C"], [0.15, 0.80]]]]},
        {"reference": "Three-phase (2 null) w/ HYPERPLANE", "comment": "Doesn't plot", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.05, 0.85]], ["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [None, None]], ["GAMMA", ["B", "C"], [0.10, 0.85]]]]},
        {"reference": "Three-phase (3 null) w/ HYPERPLANE", "comment": "Doesn't plot", "components": ["A", "B", "C"], "phases": ["ALPHA", "BETA", "GAMMA"], "conditions": {"P": 101325,"T": 1000}, "output": "ZPF", "values": [[["__HYPERPLANE__", ["B", "C"], [0.05, 0.90]], ["ALPHA", ["B", "C"], [None, None]], ["BETA", ["B", "C"], [None, None]], ["GAMMA", ["B", "C"], [None, None]]]]},
    ])

    comps = ["A", "B", "C"]
    phases = ["ALPHA", "BETA", "GAMMA"]
    conds = {v.P: 101325, v.T: 1000, v.X('B'): (0, 1, 0.01), v.X('C'): (0, 1, 0.01)}
    fig, ax = plt.subplots(subplot_kw=dict(projection="triangular"))
    dataplot(comps, phases, conds, datasets_db, ax=ax)
    # fig.savefig('test_comprehensive_ternary_dataplot_test.png', bbox_inches="tight")
    plt.close(fig)


def test_plot_interaction_runs(datasets_db):
    """Test that plot_interaction runs without an error."""
    dbf = Database(CU_MG_TDB)
    comps = ['CU', 'MG', 'VA']
    config = (('CU', 'MG'), ('VA'))

    # Plot HM_MIX without datasets
    fig = plt.figure()  # explictly do NOT pass axes to make sure they are created
    ax = plot_interaction(dbf, comps, 'FCC_A1', config, 'HM_MIX')
    # fig.savefig('test_plot_interaction_runs-figure-HM_MIX-no_datasets.png')
    plt.close(fig)

    # HM_MIX with a dataset
    datasets_db.insert(CU_MG_HM_MIX_SINGLE_FCC_A1)
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_interaction(dbf, comps, 'FCC_A1', config, 'HM_MIX', datasets_db, ax=ax)
    # fig.savefig('test_plot_interaction_runs-figure-HM_MIX.png')
    plt.close(fig)

    # Plot SM_MIX where the datasets have no data
    fig = plt.figure()
    ax = fig.add_subplot()
    plot_interaction(dbf, comps, 'FCC_A1', config, 'SM_MIX', datasets_db, ax=ax)
    # fig.savefig('test_plot_interaction_runs-figure-SM_MIX.png')
    plt.close(fig)


def test_plot_endmember_runs(datasets_db):
    """Test that plot_endmember runs without an error."""
    dbf = Database(CU_MG_TDB)
    comps = ['CU', 'MG', 'VA']
    config = ('CU', 'MG')

    endmember_dataset = {
        "components": ["CU", "MG", "VA"],
        "phases": ["CUMG2"],
        "solver": {
            "sublattice_site_ratios": [1, 2],
            "sublattice_configurations": [["CU", "MG"]],
            "mode": "manual"
        },
        "conditions": {"P": 101325, "T": [300, 400]},
        "output": "HM_FORM",
        "values":   [[[-10000], [-11000]]], "reference": "FAKE DATA",
    }
    datasets_db.insert(endmember_dataset)

    fig = plt.figure()  # explictly do NOT pass axes to make sure they are created
    ax = plot_endmember(dbf, comps, 'CUMG2', config, 'HM_FORM', datasets_db)
    # fig.savefig('test_plot_endmember_runs-figure-HM_FORM.png')
    plt.close(fig)
