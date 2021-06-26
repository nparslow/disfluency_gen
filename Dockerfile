FROM python:3.8 as base

ENV REPO_ROOT="/opt/disfluency_generator"
WORKDIR $REPO_ROOT

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY pyproject.toml .
COPY setup.cfg .
COPY src src
COPY README.md .
COPY test test
COPY run_tests.sh .

RUN python -m pip install .


