FROM jupyter/datascience-notebook

# Volumues for scripts
VOLUME [ "/usr/local/bin/start-notebook.d" ]
VOLUME [ "/usr/local/bin/before-notebook.d" ]
VOLUME [ "/home/workspace"]

# Copy fedbiomed library
COPY --chown=$NB_USER:$NB_GID .             /home/fed/.
COPY --chown=$NB_USER:$NB_GID ./notebooks/. /home/$NB_USER/notebooks/.

# Allows multikernel in notebook
RUN conda install -c conda-forge nb_conda_kernels

# This commands creates env but does not activate as default environment
RUN conda env update -f /home/fed/envs/development/docker/notebook/build_files/fedbiomed-researcher.yaml

# Set PYTHONPATH to fedbiomed directory
ENV PYTHONPATH '/home/fed'

# register the conda env to the notebook
RUN conda run -n fedbiomed-researcher ipython kernel install --user --name=fedbiomed-researcher

# automatically start shell inside this conda env
SHELL ["conda", "run", "-n", "fedbiomed-researcher", "/bin/bash", "-c"]


EXPOSE 8888
