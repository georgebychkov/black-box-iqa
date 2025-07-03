FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/gAC5XrzMm/resnet50-19c8e357.pth  https://titan.gml-team.ru:5003/fsdownload/gAC5XrzMm/resnet50-19c8e357.pth \
 && rm resnet50-19c8e357.pth.1
 
RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/4Zrsf0ktL/CKDN_model_best-38b27dc6.pth  https://titan.gml-team.ru:5003/fsdownload/4Zrsf0ktL/CKDN_model_best-38b27dc6.pth \
 && rm CKDN_model_best-38b27dc6.pth.1
 
 
COPY IQA-PyTorch /IQA-PyTorch
COPY ckdn/patches/ckdn.patch /IQA-PyTorch
RUN cd /IQA-PyTorch && unix2dos ckdn.patch && patch -Np1 < ckdn.patch
RUN cd IQA-PyTorch && pip install -r requirements.txt && python setup.py develop

COPY ckdn/test.py /

COPY ckdn/src /src
COPY ckdn/config.json /src