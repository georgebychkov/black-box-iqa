FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN set -e; \
 L=https://titan.gml-team.ru:5003/fsdownload/PbxEuuK5N/resnet50_a1_0-14fe96d1.pth; \
 wget --backups=1 -nv "$L" "$L"; rm "$(basename "$L").1"; \
 L=https://titan.gml-team.ru:5003/fsdownload/EqhzkmHVD/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz; \
 wget --backups=1 -nv "$L" "$L"; rm "$(basename "$L").1"; \
 L=https://titan.gml-team.ru:5003/fsdownload/lugmfBBGP/AHIQ_vit_p8_epoch33-da3ea303.pth; \
 wget --backups=1 -nv "$L" "$L"; rm "$(basename "$L").1";

COPY IQA-PyTorch /IQA-PyTorch
COPY ahiq/patches/ahiq.patch /IQA-PyTorch
RUN cd /IQA-PyTorch \
 && unix2dos ahiq.patch && patch -Np1 < ahiq.patch \
 && pip install -r requirements.txt && python setup.py develop

COPY ahiq/src /src

COPY ahiq/test.py /

COPY ahiq/config.json /src