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
    packages=['espei', 'espei.error_functions', 'espei.parameter_selection'],
    package_data={
        'espei': ['input-schema.yaml']
    },
    license='MIT',
    long_description=readme('README.rst'),
    url='https://espei.org/',
    install_requires=[
        'numpy',
        'scipy',
        'sympy<=1.1',
        'six',
        'dask[complete]>=0.18',
        'distributed',
        'tinydb>=3',
        'scikit-learn',
        'emcee',
        'pycalphad>=0.7',
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

        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    entry_points={'console_scripts': [
                  'espei = espei.espei_script:main']}

)
