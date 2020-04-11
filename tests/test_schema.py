"""
Tests for input file validation
"""

import pytest
import yaml
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


NULL_OUTPUTS_YAML = """
system:
  phase_models: phases.json
  datasets: ds-path
output:
  output_db:  mcmc.tdb
  verbosity:  0
  tracefile:  null
  probfile:   null
  logfile:    null
mcmc:
  iterations: 100
  scheduler: null
  input_db: dft.tdb
"""


def test_input_yaml_valid_for_full_run():
    """A minimal full run input file should validate"""
    get_run_settings(FULL_RUN_DICT)


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
    # parameters are popped off in order to assert that no additional parameters
    # are added and none are not tested.
    d = get_run_settings(FULL_RUN_DICT)
    assert d.get('output') is not None
    assert d['output'].pop('verbosity') == 0
    assert d['output'].pop('output_db') == 'out.tdb'
    assert d['output'].pop('logfile') is None
    assert d['output'].pop('tracefile') == 'trace.npy'
    assert d['output'].pop('probfile') == 'lnprob.npy'
    assert len(d['output']) == 0
    assert d['generate_parameters'].pop('ridge_alpha') is None
    assert d['generate_parameters'].pop('aicc_penalty_factor') is None
    assert len(d['generate_parameters']) == 2
    assert d['mcmc'].pop('save_interval') == 1
    assert d['mcmc'].pop('scheduler') == 'dask'
    assert d['mcmc'].pop('chains_per_parameter') == 2
    assert d['mcmc'].pop('chain_std_deviation') == 0.1
    assert d['mcmc'].pop('deterministic') is True
    assert d['mcmc'].pop('approximate_equilibrium') is False
    assert d['mcmc'].pop('data_weights') == {'ACR': 1.0, 'CPM': 1.0, 'HM': 1.0, 'SM': 1.0, 'ZPF': 1.0}
    assert d['mcmc'].pop('prior') == {'name': 'zero'}
    assert len(d['mcmc']) == 1


def test_chains_per_parameter_read_correctly():
    """The chains per parameter option should take effect when passed."""
    d = {k: v for k, v in MCMC_RUN_DICT.items()}
    d['mcmc']['chains_per_parameter'] = 6
    parsed_settings = get_run_settings(d)
    assert parsed_settings['mcmc']['chains_per_parameter'] == 6

    d['mcmc']['chains_per_parameter'] = 5
    with pytest.raises(ValueError):
        get_run_settings(d)


def test_SR2016_refdata():
    d = {k: v for k, v in GEN_PARAMS_DICT.items()}
    d['generate_parameters']['ref_state'] = 'SR2016'
    parsed_settings = get_run_settings(d)
    assert parsed_settings['generate_parameters']['ref_state'] == 'SR2016'


def test_nullable_arguments_are_all_nullable():
    nullable_dict = yaml.safe_load(NULL_OUTPUTS_YAML)
    null_settings = get_run_settings(nullable_dict)
    assert null_settings['output']['tracefile'] is None
    assert null_settings['output']['logfile'] is None
    assert null_settings['output']['probfile'] is None
    assert null_settings['mcmc']['scheduler'] is None
