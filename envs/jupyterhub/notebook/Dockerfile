FROM jupyter/datascience-notebook
EXPOSE 8888

VOLUME [ "/usr/local/bin/start-notebook.d" ]
VOLUME [ "/usr/local/bin/before-notebook.d" ]
VOLUME [ "/home/notebook" ]
VOLUME [ "/home/samples" ]
#VOLUME [ "/home/jovyan/data" ]

# Copy fedbiomed library
COPY ./fedbiomed /home/fed/fedbiomed 
COPY ./envs /home/fed/envs
COPY ./scripts /home/fed/scripts

# Set PYTHONPATH to fedbiomed directory
ENV PYTHONPATH '${PYTHONPATH}:/home/fed/'


# Install required packages
# RUN /opt/conda/bin/conda install -y paho-mqtt=1.5.1 \ 
#         pyyaml colorama nose tinydb=4.4.0 tabulate \
#         validators=0.18.2 \
#         nbstripout joblib=1.0.1 pytest requests && \
#     /opt/conda/bin/pip install torchvision==0.9.0 \
#         torch==1.8.0 \
#         JSON-log-formatter \
#         persist-queue==0.5.1

# Create ENV Variables for fedbiomed.researcher.environ 
ENV UPLOADS_URL 'http://138.96.220.179:8844/upload/'
ENV MQTT_BROKER '138.96.220.179'
ENV MQTT_BROKER_PORT '1883'


# Created env but does not activate as default environment
#RUN /home/fed/scripts/configure_conda
#RUN source /home/fed/scripts/fedbiomed_environment researcher


# Tested and got permission error
#RUN /opt/conda/bin/conda env update --file /home/fed/conda/fedbiomed-researcher.yaml
#RUN /opt/conda/bin/conda activate fedbiomed-researcher



USER $NB_UID
