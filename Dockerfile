# Work-in-progress for deploying fiting code
FROM richardotis/pycalphad-base:linux-python35
# pymc seems to need mkl...?
RUN conda install -n condaenv -y mkl scikit-learn pymc bokeh && \
    pip install git+git://github.com/pycalphad/pycalphad@develop && \
    pip install tinydb==2.4 && \
    conda install -y -n condaenv libgfortran gcc && \
    conda clean -tipsy && rm -Rf /tmp/* # 1/8/17
COPY paramselect.py /work/paramselect.py
COPY fit.py /work/fit.py
COPY input.json /work/input.json
COPY Al-Ni/input-json /work/Al-Ni
