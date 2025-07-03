FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/W3V2Zm1kl/inception_v3_google-0cc3c7bd.pth  https://titan.gml-team.ru:5003/fsdownload/W3V2Zm1kl/inception_v3_google-0cc3c7bd.pth \
 && rm inception_v3_google-0cc3c7bd.pth.1

COPY src /src

COPY patches/fid.patch /src

RUN cd /src && dos2unix piqa/fid.py && patch -Np1 < fid.patch

RUN pip install -U setuptools
RUN cd /src && pip install .

COPY model.py /src
COPY config.json /src
