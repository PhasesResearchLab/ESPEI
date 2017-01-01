# Work-in-progress for deploying fiting code
FROM richardotis/pycalphad-base:linux-python35
RUN conda install -n condaenv -y scikit-learn && \
    pip install git+git://github.com/pycalphad/pycalphad@develop && \
    conda clean -tipsy && rm -Rf /tmp/*
ADD paramselect.py /work/paramselect.py