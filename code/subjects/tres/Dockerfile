FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/p4IKpfsTK/TReS_live_1_2021.pth  https://titan.gml-team.ru:5003/fsdownload/p4IKpfsTK/TReS_live_1_2021.pth \
 && rm TReS_live_1_2021.pth.1
 
RUN pip install scipy tqdm openpyxl

COPY src /src

COPY patches/tres.patch /src

RUN cd /src && unix2dos tres.patch && patch -Np1 < tres.patch


COPY test.py /

COPY model.py /src
COPY config.json /src
