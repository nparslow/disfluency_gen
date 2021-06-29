
key idea comes from this work:
https://ieeexplore.ieee.org/document/9053226

Goal is to learn a translation model:
prompt text -> disfluent read aloud version

(and eventually - apply the model to new prompts to improve an ASR LM)

# Usage:

## Requirements:
docker usage is currently set up for GPU only, cuda 10 or 11
however outside docker, it can be run with python3
`pip3 install -r requirements.txt`
and can be installed as a module if you like:
`pip3 install .`

## Data:
1. Download a dataset, currently it's just set up for the 'LetsRead' Corpus https://lsi.co.it.pt//spl/letsreaddb.html
2. unzip and place in the data directory (e.g. data/LetsReadDB)

## Jupyter usage:
`./run_in_docker.sh jupyter`
or outside docker:
`jupyter notebook`

then open up the notebooks/train_and_test.ipynb notebook

## Script version:
`./run_in_docker.sh python -m disfluency_generator.train_model`

#Next steps:
sentence piece - required to generate sub-word units e.g. false starts
break up prompts - the network fails for longer prompts (prob due to low training data)
top k sampling - generate many outputs


Notes:
Handy for checking if gpu working in tf:
import tensorflow as tf
print(tf.test.is_gpu_available())

