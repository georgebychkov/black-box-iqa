FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/6KzEo0yMc/CNNIQA-LIVE.pth  https://titan.gml-team.ru:5003/fsdownload/6KzEo0yMc/CNNIQA-LIVE.pth \
 && rm CNNIQA-LIVE.pth.1
 
RUN pip install scipy h5py PyYAML Pillow pytorch-ignite tensorboardX

COPY src /src

COPY test.py /

COPY model.py /src