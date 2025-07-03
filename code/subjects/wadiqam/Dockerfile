FROM python:2

WORKDIR /src
COPY src/requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY src ./
COPY patches ./

RUN patch -Np1 < 01.patch
CMD python test.py
