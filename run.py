from pycalphad.fitting import build_pymc_model, plot_results
from pycalphad import Database
import pymc
import numpy as np
import sys
import os
import time
import glob
from sumatra.projects import load_project
from sumatra.parameters import build_parameters

def main(parameters):
  input_database = Database(parameters['input_database'])
  dataset_names = sorted(glob.glob(parameters['data_path']))
  params = []
  for pname, paramdist in parameters['parameters'].items():
    dist = getattr(pymc, paramdist['dist'].pop())
    params.append(dist(pname, **paramdist))
  mod, datasets = build_pymc_model(input_database, dataset_names, params)
  trace_path = os.path.join(parameters['results_path'],
			    'output-traces', parameters['sumatra_label'])
  MDL = pymc.MCMC(mod, db='hdf5', dbname=trace_path+'traces.hdf5',
                  dbcomplevel=4, dbcomplib='bzip2')
  MDL.sample(**parameters['mcmc'])

parameter_file = sys.argv[1]
parameters = build_parameters(parameter_file)

project = load_project()
record = project.new_record(parameters=parameters,
                            main_file=__file__,
                            reason="reason for running this simulation")
parameters.update({"sumatra_label": record.label})
start_time = time.time()

main(parameters)

record.duration = time.time() - start_time
record.output_data = record.datastore.find_new_data(record.timestamp)
project.add_record(record)

project.save()