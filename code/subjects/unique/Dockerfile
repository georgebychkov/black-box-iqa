FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

COPY download.sh /
RUN bash download.sh

COPY src /src
COPY patches /src
COPY model.py /src
RUN cd src && patch -Np1 < unique.patch

COPY config.json /src
