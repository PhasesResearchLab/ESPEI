"""
Author: Jonathan Siegel

Contains methods which test the sampling and optimization methods for the chemical potential
optimization.
"""

import numpy as np
from chemical_potential_optimizer import generate_phase_samples

def test_generate_phase_samples():
  print(generate_phase_samples([3], 10))
  print(generate_phase_samples([3,2,1],10))

if __name__ == '__main__':
  test_generate_phase_samples()
