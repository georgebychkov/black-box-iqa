FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/tQ9IE8KJ2/FAST_VQA_3D_1_1.pth  https://titan.gml-team.ru:5003/fsdownload/tQ9IE8KJ2/FAST_VQA_3D_1_1.pth \
 && rm FAST_VQA_3D_1_1.pth.1
 
 
COPY src /src

COPY patches/faster-vqa.patch /src

RUN cd /src && unix2dos faster-vqa.patch && patch -Np1 < faster-vqa.patch

RUN cd /src && pip install -r requirements.txt


COPY model.py /src
COPY config.json /src
