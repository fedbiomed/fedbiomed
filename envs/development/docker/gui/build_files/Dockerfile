FROM debian:bullseye-slim as builder


VOLUME ["/app"]
VOLUME ["/data/images"]
VOLUME ["/data/csv"]

# Installl
RUN apt-get update && apt-get install -y apt-utils wget

COPY envs/development/docker/gui/build_files/entrypoint.sh /
COPY envs /fedbiomed/envs
COPY fedbiomed /fedbiomed/fedbiomed
COPY scripts /fedbiomed/scripts

   
# Install miniconda
RUN wget -q --directory-prefix=$HOME https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
        bash $HOME/Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda && \
        rm -f $HOME/Miniconda3-latest-Linux-x86_64.sh

ENV PATH=${PATH}:/miniconda/bin

WORKDIR /fedbiomed
RUN conda env update --file ./envs/development/conda/ui.yaml

EXPOSE 5000
ENTRYPOINT ["tail", "-f", "/dev/null"]