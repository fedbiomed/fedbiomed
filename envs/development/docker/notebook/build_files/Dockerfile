FROM jupyter/datascience-notebook

# Volumues for scripts
VOLUME [ "/usr/local/bin/start-notebook.d" ]
VOLUME [ "/usr/local/bin/before-notebook.d" ]
VOLUME [ "/home/workspace"]

# Copy fedbiomed library
COPY ./fedbiomed /home/fed/fedbiomed 
COPY ./envs/development/docker/notebook/build_files/requirements-conda.txt /home/fed/requirements/
COPY ./envs/development/docker/notebook/build_files/requirements-pip.txt /home/fed/requirements/
COPY ./notebooks /home/$NB_USER/notebooks/

# Set PYTHONPATH to fedbiomed directory
ENV PYTHONPATH '${PYTHONPATH}:/home/fed/'

RUN /opt/conda/bin/conda install -y --file /home/fed/requirements/requirements-conda.txt && \
        conda clean --all -f -y && \
        fix-permissions "${CONDA_DIR}" && \
        fix-permissions "/home/${NB_USER}"
RUN /opt/conda/bin/pip install -r /home/fed/requirements/requirements-pip.txt && \
        fix-permissions "${CONDA_DIR}" && \
        fix-permissions "/home/${NB_USER}"


# This commands creates env but does not activate as default environment
#RUN /home/fed/scripts/configure_conda
#RUN source /home/fed/scripts/fedbiomed_environment researcher


# Tested and got permission error
#RUN /opt/conda/bin/conda env update --file /home/fed/conda/fedbiomed-researcher.yaml
#RUN /opt/conda/bin/conda activate fedbiomed-researcher

EXPOSE 8888
