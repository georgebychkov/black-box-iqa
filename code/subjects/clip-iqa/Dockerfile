FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/82VpF0Dtv/cliq-iqa_iter_80000.pth  https://titan.gml-team.ru:5003/fsdownload/82VpF0Dtv/cliq-iqa_iter_80000.pth \
 && rm cliq-iqa_iter_80000.pth.1
 
COPY src /src

COPY requirements.txt /src
RUN cd /src && pip install -r requirements.txt

RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

COPY patches/clip-iqa.patch /src
RUN cd /src && unix2dos clip-iqa.patch && dos2unix mmedit/models/restorers/clipiqa.py && dos2unix mmedit/models/backbones/sr_backbones/coopclipiqa.py && patch -Np1 < clip-iqa.patch

COPY test.py /

COPY model.py /src



COPY config.json /src
