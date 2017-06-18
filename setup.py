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
    packages=['espei'],
    license='MIT',
    long_description=readme('README.rst'),
    url='https://espei.org/',
    install_requires=['numpy', 'scipy', 'dask', 'distributed', 'scikit-learn', 'emcee', 'pycalphad'],
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6'
    ],
    entry_points={'console_scripts': [
                  'espei = espei.fit:main']}

)