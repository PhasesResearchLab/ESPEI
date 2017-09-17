"""
Tests for input file validation
"""

import yaml
from espei import schema

def test_input_yaml_valid_for_full_run():
    pass


def test_input_yaml_valid_for_generate_parameters_only():
    pass


def test_input_yaml_valid_for_mcmc_from_tdb():
    pass


def test_input_yaml_valid_for_mcmc_from_restart():
    pass

def test_input_yaml_invalid_for_mcmc_when_input_not_defined():
    """An MCMC run must get input from generate_parameters, an input tdb, or a restart and input tdb."""
    pass

def test_correct_defaults_are_applied_from_minimal_specification():
    """"""
    pass
