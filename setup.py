from setuptools import setup
import os
import versioneer

def readme(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='espei',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brandon Bocklund',
    author_email='brandonbocklund@gmail.com',
    description='Fitting thermodynamic models with pycalphad.',
    packages=['espei', 'espei.error_functions', 'espei.parameter_selection', 'espei.optimizers'],
    package_data={
        'espei': ['input-schema.yaml']
    },
    license='MIT',
    long_description=readme('README.rst'),
    url='https://espei.org/',
    install_requires=[
        'numpy',
        'scipy',
        'sympy>=1.2',
        'dask[complete]>=2',
        'distributed>=2',
        'tinydb>=3.8',
        'scikit-learn',
        'emcee<3',
        'pycalphad>=0.8.2',
        'pyyaml',
        'cerberus',
        'bibtexparser'],
    extras_require={
        'dev': [
            'sphinx',
            'sphinx_rtd_theme',
            'pytest',
            'nose', # required only until matplotlib switches to pytest
            'mock',
            'twine',
        ],
        'mpi': [
            'mpi4py',
            'dask-mpi>=2',
        ]
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={'console_scripts': [
                  'espei = espei.espei_script:main']}

)
