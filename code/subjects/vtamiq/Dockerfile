FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/p36e47QQs/latest.pth  https://titan.gml-team.ru:5003/fsdownload/p36e47QQs/latest.pth \
 && rm latest.pth.1

 
COPY src /src

COPY patches/vtamiq.patch /src

RUN cd /src && unix2dos vtamiq.patch && patch -Np1 < vtamiq.patch

COPY pyproject.toml /

RUN pip install -e .

COPY model.py /src
COPY test.py /
COPY config.json /src
