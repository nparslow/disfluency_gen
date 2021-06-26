FROM python:3.8 as base

ENV REPO_ROOT="/opt/disfluency_generator"
WORKDIR $REPO_ROOT

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . .

RUN python -m pip install .


