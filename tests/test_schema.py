"""
Tests for input file validation
"""

import pytest
from espei.espei_script import get_run_settings

FULL_RUN_DICT = {
    'generate_parameters':
        {
            'excess_model': 'linear', 'ref_state': 'SGTE91'
        },
    'mcmc':
        {
            'iterations': 1000
        },
    'system':
        {
            'datasets': 'ds_path',
            'phase_models': 'phases.json'
        }
}

GEN_PARAMS_DICT = {
    'generate_parameters':
        {
            'excess_model': 'linear',
            'ref_state': 'SGTE91'
        },
    'system':
        {
            'datasets': 'ds_path',
            'phase_models': 'phases.json'
        }
}

MCMC_RUN_DICT = {
    'mcmc':
        {
            'iterations': 1000,
            'input_db': 'input.tdb',
        },
    'system':
        {
            'datasets': 'ds_path',
            'phase_models': 'phases.json'
        }
}

MCMC_RESTART_DICT = {
    'mcmc':
        {
            'iterations': 1000,
            'input_db': 'input.tdb',
            'restart_trace': 'restart_trace.npy'
        },
    'system':
        {
            'datasets': 'ds_path',
            'phase_models': 'phases.json'
        }
}

MCMC_NO_INPUT_DICT = {
    'mcmc':
        {
            'iterations': 1000
        },
    'system':
        {
            'datasets': 'ds_path',
            'phase_models': 'phases.json'
        }
}

MCMC_OVERSPECIFIED_INPUT_DICT = {
    'generate_parameters':
        {
            'excess_model': 'linear', 'ref_state': 'SGTE91'
        },
    'mcmc':
        {
            'iterations': 1000,
            'input_db': 'my_input.db'
        },
    'system':
        {
            'datasets': 'ds_path',
            'phase_models': 'phases.json'
        }
}


def test_input_yaml_valid_for_full_run():
    """A minimal full run input file should validate"""
    d = get_run_settings(FULL_RUN_DICT)


def test_input_yaml_valid_for_generate_parameters_only():
    """A minimal generate parameters only input file should validate"""
    d = get_run_settings(GEN_PARAMS_DICT)
    assert d.get('mcmc') is None


def test_input_yaml_valid_for_mcmc_from_tdb():
    """A minimal mcmc run from tdb input file should validate"""
    d = get_run_settings(MCMC_RUN_DICT)
    assert d.get('generate_parameters') is None


def test_input_yaml_valid_for_mcmc_from_restart():
    """A minimal mcmc run from a restart should validate"""
    d = get_run_settings(MCMC_RESTART_DICT)
    assert d.get('generate_parameters') is None


def test_input_yaml_invalid_for_mcmc_when_input_not_defined():
    """An MCMC run must get input from generate_parameters, an input tdb, or a restart and input tdb."""
    with pytest.raises(ValueError):
        get_run_settings(MCMC_NO_INPUT_DICT)


def test_input_yaml_invalid_for_mcmc_when_input_is_overspecified():
    """An MCMC run must get input from only generate_parameters or an input tdb (w/ or w/o a restart)."""
    with pytest.raises(ValueError):
        get_run_settings(MCMC_OVERSPECIFIED_INPUT_DICT)


def test_correct_defaults_are_applied_from_minimal_specification():
    """A minimal run should generate several default settings for i/o and optional settings."""
    d = get_run_settings(FULL_RUN_DICT)
    assert d.get('output') is not None
    assert d['output']['verbosity'] == 0
    assert d['output']['output_db'] == 'out.tdb'
    assert d['output']['tracefile'] == 'trace.npy'
    assert d['output']['probfile'] == 'lnprob.npy'
    assert d['generate_parameters']['ridge_alpha'] == 1e-100
    assert d['mcmc']['save_interval'] == 1
    assert d['mcmc']['scheduler'] == 'dask'
    assert d['mcmc']['chains_per_parameter'] == 2
    assert d['mcmc']['chain_std_deviation'] == 0.1
    assert d['mcmc']['deterministic'] == True


def test_chains_per_parameter_read_correctly():
    """The chains per parameter option should take effect when passed."""
    d = {k: v for k,v in MCMC_RUN_DICT.items()}
    d['mcmc']['chains_per_parameter'] = 6
    parsed_settings = get_run_settings(d)
    assert parsed_settings['mcmc']['chains_per_parameter'] == 6

    d['mcmc']['chains_per_parameter'] = 5
    with pytest.raises(ValueError):
        get_run_settings(d)

def test_SR2016_refdata():
    d = {k: v for k,v in GEN_PARAMS_DICT.items()}
    d['generate_parameters']['ref_state'] = 'SR2016'
    parsed_settings = get_run_settings(d)
    assert parsed_settings['generate_parameters']['ref_state'] == 'SR2016'
