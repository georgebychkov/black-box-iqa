FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/6uCdHlNj6/checkpoints_swa-equal-21-30.ckpt  https://titan.gml-team.ru:5003/fsdownload/6uCdHlNj6/checkpoints_swa-equal-21-30.ckpt \
 && rm checkpoints_swa-equal-21-30.ckpt.1
 
 
COPY src /src

COPY patches/conformer-bns.patch /src

RUN cd /src && unix2dos conformer-bns.patch && patch -Np1 < conformer-bns.patch

RUN cd /src && pip install -r requirements.txt

COPY test.py /

COPY model.py /src
COPY config.json /src

