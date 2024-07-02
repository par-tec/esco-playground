FROM ghcr.io/par-tec/ml-playground:py310-spacy-20231120-18-a6ad1df
RUN apt-get -y update && \
    apt-get -y install python3-pip

RUN python3 -m pip install tox
