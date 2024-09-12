FROM ubuntu:22.04
LABEL maintainer='axelc@qubit-pharmaceuticals.com'

ARG DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "--login", "-c"]

RUN apt-get update
RUN apt-get install -yqq vim wget python3 python3-pip python-is-python3
RUN apt-get install -yqq awscli

WORKDIR /codes
COPY /codes .

COPY requirements.txt .
RUN pip install -r requirements.txt
