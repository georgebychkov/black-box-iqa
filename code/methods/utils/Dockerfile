ARG METRIC_IMAGE=None
FROM ${METRIC_IMAGE} as base

#COPY --from=vg-code.gml-team.ru:4567/framework/attacks/datasets/coco:25e_shu /data /

RUN apt-get update -q 
RUN apt-get install build-essential -qqy
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install av
RUN pip3 install scipy
RUN pip3 install scikit-image
RUN pip3 install PyWavelets
RUN pip3 install IQA_pytorch
RUN pip3 install frozendict
RUN pip3 install lpips
RUN pip3 install torchmetrics
RUN pip3 install pytorch-wavelets
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install pyiqa
RUN pip3 install albumentations==1.3.1
RUN pip3 install deap


COPY ./read_dataset.py /read_dataset.py
COPY ./metrics.py /metrics.py
COPY ./bounds.json /bounds.json
COPY ./evaluate.py /evaluate.py
COPY ./fgsm_evaluate.py /fgsm_evaluate.py
COPY ./uap_evaluate.py /uap_evaluate.py
COPY ./cnn_evaluate.py /cnn_evaluate.py
COPY ./read_dataset.py /read_dataset.py
COPY ./test_results.py /test_results.py


COPY run.py /run.py

COPY ./vqmt-14.1.12839.pro-Linux.deb /vqmt-14.1.12839.pro-Linux.deb
COPY ./config.json /vqmt_config.json
RUN apt install ocl-icd-libopencl1
RUN dpkg -i /vqmt-14.1.12839.pro-Linux.deb && vqmt -activate < /vqmt_config.json
