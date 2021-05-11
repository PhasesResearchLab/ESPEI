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
