from pycalphad.fitting import build_pymc_model, plot_results
from pycalphad import Database
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
import tables
import matplotlib.pyplot as plt
import pymc
import numpy as np
import sys
import os
import time
import glob
from collections import OrderedDict


def autocorr(x):
    x = (x - x.mean()) / np.linalg.norm(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


def main(parameters):
  input_database = Database(parameters['input_database'])
  dataset_names = sorted(glob.glob(parameters['data_path']))
  params = []
  for pname, paramdist in parameters['parameters'].items():
    dist = getattr(pymc, paramdist.pop('dist'))
    params.append(dist(str(pname), **paramdist))
  mod, datasets = build_pymc_model(input_database, dataset_names, params)
  trace_path = os.path.join('Data', parameters['sumatra_label'])
  os.makedirs(trace_path)
  MDL = pymc.MCMC(mod, db='hdf5', dbname=str(os.path.join(trace_path, 'traces.hdf5')),
                  dbcomplevel=4, dbcomplib='bzip2')
  MDL.sample(**parameters['mcmc'])
  MDL.db.close()


def analyze(parameters):
  trace_file = str(os.path.join('Data', parameters['sumatra_label'], 'traces.hdf5'))
  image_path = os.path.join('Data', parameters['sumatra_label'])
  data = tables.open_file(trace_file, mode='r')
  data_dict = OrderedDict()
  parnames = [x for x in data.root.chain0.PyMCsamples.colnames
	      if not x.startswith('Metropolis') and x != 'deviance']
  for param in sorted(parnames):
      data_dict[param] = np.asarray(data.root.chain0.PyMCsamples.read(field=param), dtype='float')
      for param, trace in data_dict.items():
	  figure = plt.figure()
	  figure.gca().plot(autocorr(trace))
	  figure.gca().set_title(param+' Autocorrelation')
	  figure.savefig(str(os.path.join(image_path, param+'-acf.eps')))
	  figure.close()

parameter_file = sys.argv[1]
parameters = build_parameters(parameter_file)

project = load_project()
record = project.new_record(parameters=parameters,
                            main_file=__file__,
                            reason="reason for running this simulation")
record.tags = set([str(i) for i in parameters['phases']+parameters['components']])
parameters.update({"sumatra_label": record.label})
start_time = time.time()

main(parameters)
analyze(parameters)

record.duration = time.time() - start_time
record.output_data = record.datastore.find_new_data(record.timestamp)
project.add_record(record)

project.save()