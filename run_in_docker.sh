#!/usr/bin/env bash

set -e

usage() {
  echo "
    Usage: $0 [-d data_path] [script_to_run]
                       e.g. $0 -d data/LetsReadCorpus bash

           script_to_run: put 'test' to run tests and code coverage
                          put 'dev' or 'bash' to open a bash terminal in the container

                          if no script is entered, will only build the container
       " 1>&2
  exit 1;
}

DATA_DIR="$(pwd)/data"
while getopts "d:h" opt; do
  case ${opt} in
    d)
      DATA_DIR=$OPTARG
      ;;
    h)
      usage
      ;;
    *)
      echo "Invalid option: $OPTARG" 1>&2
      usage
  esac
done
shift $((OPTIND -1))

script_to_run=${1}

case ${script_to_run} in
  "test")
    script_to_run="./run_tests.sh"
    ;;
  "dev")
    script_to_run="bash"
    ;;
esac

INTERACTIVE=
if [ "${script_to_run}" == "bash" ]; then
  INTERACTIVE="-it"
fi

IMAGE_NAME=disfluency_generator

echo "/* Building Docker Image */"

docker build -t ${IMAGE_NAME} .

docker build -t ${IMAGE_NAME} .

if [ -n "${script_to_run}" ]; then

  echo "/* Running Docker Image */"

  docker run --rm ${INTERACTIVE} \
         -v "${DATA_DIR}:/opt/dataset-preparation/data" \
         "${IMAGE_NAME}" ${script_to_run}
fi

