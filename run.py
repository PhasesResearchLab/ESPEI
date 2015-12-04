from pycalphad.fitting import build_pymc_model, plot_results, setup_dataset, Dataset
from pycalphad import Database
from sumatra.projects import load_project
from sumatra.parameters import build_parameters
from sumatra.datastore.filesystem import DataFile
from corner import corner
import tables
import matplotlib
matplotlib.style.use('bmh')
import matplotlib.pyplot as plt
import pymc
from pymc.database.base import batchsd
from pymc import utils
import numpy as np
import sys
import os
import shutil
import time
import glob
import csv
from collections import OrderedDict, defaultdict
from itertools import chain


def autocorr(x):
    x = (x - x.mean()) / np.linalg.norm(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

output_files = []


def main(parameters, seed):
    np.random.seed(seed)
    input_database = Database(parameters['input_database'])
    dataset_names = sorted(glob.glob(parameters['data_path']))
    params = []
    for pname, paramdist in parameters['parameters'].items():
        paramdist = paramdist.copy()  # don't want to modify original
        paramdist.pop('compare')  # don't pass this as a kwarg
        dist = getattr(pymc, paramdist.pop('dist'))
        params.append(dist(str(pname), **paramdist))
    mod, datasets = build_pymc_model(input_database, dataset_names, params)
    trace_path = os.path.join('Data', parameters['sumatra_label'])
    os.makedirs(trace_path)
    mdl = pymc.MCMC(mod, db='hdf5', dbname=str(os.path.join(trace_path, 'traces.h5')),
                    dbcomplevel=4, dbcomplib='bzip2')
    mdl.sample(**parameters['mcmc'])
    mdl.db.close()
    output_files.append(os.path.join(trace_path, 'traces.h5'))
    return datasets


def analyze(parameters, datasets):
    image_path = os.path.join('Data', parameters['sumatra_label'])
    # Save traces
    trace_file = str(os.path.join('Data', parameters['sumatra_label'], 'traces.h5'))
    data_dict = OrderedDict()
    os.makedirs(os.path.join(image_path, 'acf'))
    with tables.open_file(trace_file, mode='r') as data:
        parnames = [x for x in data.root.chain0.PyMCsamples.colnames
                    if not x.startswith('Metropolis') and x != 'deviance']
        for param in sorted(parnames):
            data_dict[param] = np.asarray(data.root.chain0.PyMCsamples.read(field=param), dtype='float')
    for param, trace in data_dict.items():
        figure = plt.figure()
        figure.gca().plot(autocorr(trace))
        figure.gca().set_title(param+' Autocorrelation')
        figure.savefig(str(os.path.join(image_path, 'acf', param+'.png')))
        plt.close(figure)
        output_files.append(str(os.path.join(image_path, 'acf', param+'.png')))

    data = np.vstack(list(data_dict.values())).T
    data_truths = [parameters.as_dict()['parameters'][key].get('compare', None) for key in data_dict.keys()]
    figure = corner(data, labels=list(data_dict.keys()),
                    quantiles=[0.16, 0.5, 0.84],
                    truths=data_truths,
                    show_titles=True, title_args={"fontsize": 40}, rasterized=True)
    figure.savefig(str(os.path.join(image_path, 'cornerplot.png')))
    output_files.append(str(os.path.join(image_path, 'cornerplot.png')))
    plt.close(figure)
    # Write CSV file with parameter summary (should be close to pymc's format)
    with open(str(os.path.join(image_path, 'parameters.csv')), 'w') as csvfile:
        fieldnames = ['Parameter', 'Mean', 'SD', 'Lower 95% HPD', 'Upper 95% HPD',
                      'MC error', 'q2.5', 'q25', 'q50', 'q75', 'q97.5']
        writer = csv.DictWriter(csvfile, fieldnames)
        writer.writeheader()
        for parname, trace in data_dict.items():
            qxx = utils.quantiles(trace, qlist=(2.5, 25, 50, 75, 97.5))
            q2d5, q25, q50, q75, q975 = qxx[2.5], qxx[25], qxx[50], qxx[75], qxx[97.5]
            lower_hpd, upper_hpd = utils.hpd(trace, 0.05)
            row = {
                'Parameter': parname,
                'Mean': trace.mean(0),
                'SD': trace.std(0),
                'Lower 95% HPD': lower_hpd,
                'Upper 95% HPD': upper_hpd,
                'MC error': batchsd(trace, min(len(trace), 100)),
                'q2.5': q2d5, 'q25': q25, 'q50': q50, 'q75': q75, 'q97.5': q975
            }
            writer.writerow(row)
    # Generate comparison figures
    os.makedirs(os.path.join(image_path, 'results'))
    input_database = Database(parameters['input_database'])
    compare_databases = {key: Database(value) for key, value in parameters['compare_databases'].items()}
    idx = 1
    for fig in plot_results(input_database, datasets, data_dict, databases=compare_databases):
        fig.savefig(str(os.path.join(image_path, 'results', 'Figure{}.png'.format(idx))))
        output_files.append(str(os.path.join(image_path, 'results', 'Figure{}.png'.format(idx))))
        plt.close(fig)
        idx += 1

parameter_file = sys.argv[1]
parameters = build_parameters(parameter_file)


project = load_project()
record = project.new_record(parameters=parameters,
                            main_file=__file__,
                            reason="Single-phase fitting tests using Sumatra")
# Add some tags related to the phases and components present
record.tags = set([str(i) for i in parameters['phases']+parameters['components']])
record.tags |= {'CALPHAD'}
# Fix a random seed in case we need it again
seed = parameters.as_dict().get('seed', np.random.randint(0, 1e5))
parameters.update({"sumatra_label": record.label, "seed": seed})
start_time = time.time()

datasets = main(parameters, seed)
analyze(parameters, datasets)

record.duration = time.time() - start_time
record.input_data = []
input_path = os.path.join('Data', parameters['sumatra_label'], 'input')
os.makedirs(input_path)

mime_exts = defaultdict(lambda: 'text/plain')
mime_exts.update({'csv': 'text/csv',
                  'h5': None})

for inp in [parameters['input_database']] + sorted(glob.glob(parameters['data_path'])):
    # copy2 preserves most metadata
    shutil.copy2(str(inp), input_path)
    record.input_data.append(DataFile(os.path.join(str(parameters['sumatra_label']), 'input',
                                                   os.path.basename(str(inp))),
                                      project.data_store).generate_key())

for outp in output_files:
    record.output_data.append(DataFile(outp,
                                       project.data_store).generate_key())

# Workaround for broken mime type detection
for fkey in chain(record.input_data, record.output_data):
    fkey.metadata['mimetype'] = fkey.metadata['mimetype'] or mime_exts[os.path.splitext(fkey.path)[-1]]
project.add_record(record)

project.save()
