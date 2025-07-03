FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/gS2g4BqXO/NAR_student_cross_dataset.pth  https://titan.gml-team.ru:5003/fsdownload/gS2g4BqXO/NAR_student_cross_dataset.pth \
 && rm NAR_student_cross_dataset.pth.1

COPY src /src

COPY patches/cvrkd-iqa.patch /src
RUN cd /src && dos2unix cvrkd-iqa.patch && patch -Np1 < cvrkd-iqa.patch

RUN pip install einops

COPY model.py /src
COPY config.json /src


