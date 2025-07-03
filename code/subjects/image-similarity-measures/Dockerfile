FROM python:3

COPY ./src /src
WORKDIR /src

RUN apt-get update -qq && apt-get install -qqy libfftw3-dev libgl1

#RUN bash setup.sh
