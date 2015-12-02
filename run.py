from pycalphad.fitting import build_pymc_model, plot_results
from pycalphad import Database
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
from corner import corner
import tables
import matplotlib.pyplot as plt
import pymc
import numpy as np
import sys
import os
import time
import glob
from collections import OrderedDict
from itertools import chain


def autocorr(x):
    x = (x - x.mean()) / np.linalg.norm(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


def main(parameters, seed):
  np.random.seed(seed)
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
  data_dict = OrderedDict()
  with tables.open_file(trace_file, mode='r') as data:
    parnames = [x for x in data.root.chain0.PyMCsamples.colnames
		if not x.startswith('Metropolis') and x != 'deviance']
    for param in sorted(parnames):
      data_dict[param] = np.asarray(data.root.chain0.PyMCsamples.read(field=param), dtype='float')
      for param, trace in data_dict.items():
	figure = plt.figure()
	figure.gca().plot(autocorr(trace))
	figure.gca().set_title(param+' Autocorrelation')
	figure.savefig(str(os.path.join(image_path, param+'-acf.png')))
	plt.close(figure)

  data = np.vstack(chain([i for i in data_dict.values()])).T
  figure = corner(data, labels=list(data_dict.keys()),
		  quantiles=[0.16, 0.5, 0.84],
		  truths=[-1622407.75, 16.212965, 73417.798, -34.914168, 33471.014, -9.8373558,
			  -30758.01, 10.25267, 0.52, -1112, 1745, -22212.8931, 4.39570389],
		  show_titles=True, title_args={"fontsize": 40}, rasterized=True)
  figure.savefig(str(os.path.join(image_path, 'cornerplot.png')))
  plt.close(figure)

parameter_file = sys.argv[1]
parameters = build_parameters(parameter_file)

project = load_project()
record = project.new_record(parameters=parameters,
                            main_file=__file__,
                            reason="reason for running this simulation")
# Add some tags related to the phases and components present
record.tags = set([str(i) for i in parameters['phases']+parameters['components']])
# Fix a random seed in case we need it again
seed = parameters.as_dict().get('seed', np.random.randint(0, 1e5))
parameters.update({"sumatra_label": record.label, "seed": seed})
start_time = time.time()

main(parameters, seed)
analyze(parameters)

record.duration = time.time() - start_time
record.output_data = record.datastore.find_new_data(record.timestamp)
project.add_record(record)

project.save()