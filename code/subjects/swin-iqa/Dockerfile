FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/XuZuU6F9U/swin-iqa.pt  https://titan.gml-team.ru:5003/fsdownload/XuZuU6F9U/swin-iqa.pt \
 && rm swin-iqa.pt.1
 
RUN pip install timm einops scipy

COPY src /src

COPY model.py /src
COPY config.json /src
