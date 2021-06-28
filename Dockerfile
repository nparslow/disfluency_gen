FROM tensorflow/tensorflow:2.4.2-gpu-jupyter AS base 

ENV REPO_ROOT="/opt/disfluency_generator"
WORKDIR $REPO_ROOT

# cuda 11 solution
RUN ln /usr/local/cuda/lib64/libcusolver.so.10 /usr/local/cuda/lib64/libcusolver.so.11

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY pyproject.toml .
COPY setup.cfg .
COPY src src
COPY README.md .
COPY test test
COPY run_tests.sh .

RUN python -m pip install .

