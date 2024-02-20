FROM python:3.11-bookworm
RUN apt-get -y update && \
    apt-get -y install python3-pip

RUN python3 -m pip install tox
