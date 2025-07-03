FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/KkEgCgJ9y/resnet50-0676ba61.pth  https://titan.gml-team.ru:5003/fsdownload/KkEgCgJ9y/resnet50-0676ba61.pth \
 && rm resnet50-0676ba61.pth.1
 
RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/qZzcZYNdR/CONTRIQUE_checkpoint25.tar  https://titan.gml-team.ru:5003/fsdownload/qZzcZYNdR/CONTRIQUE_checkpoint25.tar \
 && rm CONTRIQUE_checkpoint25.tar.1
 
RUN pip install scikit-learn

COPY src /src

COPY test.py /
