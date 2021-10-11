FROM jupyter/datascience-notebook
EXPOSE 8888

VOLUME [ "/usr/local/bin/start-notebook.d" ]
VOLUME [ "/usr/local/bin/before-notebook.d" ]
VOLUME [ "/home/notebook" ]
#VOLUME [ "/home/samples" ]
#VOLUME [ "/home/jovyan/data" ]

# Copy fedbiomed library
COPY ./fedbiomed /home/fed/fedbiomed 
COPY ./envs /home/fed/envs
COPY ./scripts /home/fed/scripts
COPY ./envs/development/docker/notebook/build_files/requirements-conda.txt /home/fed/requirements/
COPY ./envs/development/docker/notebook/build_files/requirements-pip.txt /home/fed/requirements/
COPY ./notebooks /home/notebooks/

# Set PYTHONPATH to fedbiomed directory
ENV PYTHONPATH '${PYTHONPATH}:/home/fed/'

# RUN /opt/conda/bin/conda install -y --file /home/fed/requirements/requirements-conda.txt && \
#         conda clean --all -f -y && \
#         fix-permissions "${CONDA_DIR}" && \
#         fix-permissions "/home/${NB_USER}"
# RUN /opt/conda/bin/pip install -r /home/fed/requirements/requirements-pip.txt&& \
#         fix-permissions "${CONDA_DIR}" && \
#         fix-permissions "/home/${NB_USER}"


# Created env but does not activate as default environment
#RUN /home/fed/scripts/configure_conda
#RUN source /home/fed/scripts/fedbiomed_environment researcher


# Tested and got permission error
#RUN /opt/conda/bin/conda env update --file /home/fed/conda/fedbiomed-researcher.yaml
#RUN /opt/conda/bin/conda activate fedbiomed-researcher



USER $NB_UID
