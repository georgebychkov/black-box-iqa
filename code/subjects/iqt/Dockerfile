FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/k8uuOexv8/inceptionresnetv2-520b38e4.pth  https://titan.gml-team.ru:5003/fsdownload/k8uuOexv8/inceptionresnetv2-520b38e4.pth \
 && rm inceptionresnetv2-520b38e4.pth.1

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/ucg16ta82/iqt_PIPAL.pth  https://titan.gml-team.ru:5003/fsdownload/ucg16ta82/iqt_PIPAL.pth \
 && rm iqt_PIPAL.pth.1

COPY src /src

COPY patches/iqt.patch /src
RUN cd /src && dos2unix iqt.patch && dos2unix model/backbone.py && patch -Np1 < iqt.patch

COPY model_iqt.py /src
COPY config.json /src

RUN pip install einops