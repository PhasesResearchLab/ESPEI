# Work-in-progress for deploying fiting code
FROM richardotis/pycalphad-base:linux-python35
# pymc seems to need mkl...?
RUN conda install -n condaenv -y mkl scikit-learn pymc bokeh && \
    conda remove -y --offline -n condaenv tinydb gmpy2 && \
    pip install git+git://github.com/pycalphad/pycalphad@develop && \
    pip install tinydb==2.4 && \
    conda clean -tipsy && rm -Rf /tmp/* # 1/9/2017 4:18pm
COPY paramselect.py /work/paramselect.py
COPY fit.py /work/fit.py
COPY input.json /work/input.json
COPY Al-Ni/input-json /work/Al-Ni