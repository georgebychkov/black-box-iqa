FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/Rb8ocvSRn/alex_shift_tolerant.pth  https://titan.gml-team.ru:5003/fsdownload/Rb8ocvSRn/alex_shift_tolerant.pth \
 && rm alex_shift_tolerant.pth.1
 
 
COPY src /src
RUN pip install /src

COPY test.py /
COPY config.json /src