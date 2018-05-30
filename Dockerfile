From ubuntu:17.10

RUN apt update
RUN apt install gcc libc-dev g++ libssl-dev build-essential cmake make wget git -y
RUN apt install python3.6 python3.6-dev -y
RUN wget --no-verbose https://bootstrap.pypa.io/get-pip.py
RUN python3.6 get-pip.py

WORKDIR /service
RUN export C_INCLUDE_PATH=/usr/include
RUN pip3 install --upgrade pip
COPY ./requirements.txt /service/requirements.txt
RUN pip3 install -r /service/requirements.txt

COPY ./ /service
WORKDIR /service
CMD python3.6 -m src.main

