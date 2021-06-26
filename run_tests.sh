#!/usr/bin/env bash

# see "coverage run -h" and "coverage report -h" for more options:
coverage run --source=src -m unittest discover
coverage report --skip-empty

