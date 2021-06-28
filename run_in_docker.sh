#!/usr/bin/env bash

set -e

usage() {
  echo "
    Usage: $0 [-d data_path] [script_to_run]
                       e.g. $0 -d data/LetsReadCorpus bash

           script_to_run: 'test' to run tests and code coverage
                          'dev' or 'bash' to open a bash terminal in the container

			  'jupyter' to open a jupyter session, port 8082

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
PORT=
INTERACTIVE=
NOTEBOOK_PATH=
case ${script_to_run} in
  "test"|"tests")
    script_to_run="./run_tests.sh"
    ;;
  "dev"|"bash")
    script_to_run="bash"
    INTERACTIVE="-it"
    ;;
  "jupyter")
    PORT=8082
    NOTEBOOK_PATH="$(pwd)/notebooks"
    script_to_run="jupyter notebook --port=$PORT --no-browser --ip=0.0.0.0 --allow-root"
    ;;
esac

IMAGE_NAME=disfluency_generator

echo "/* Building Docker Image */"

docker build -t ${IMAGE_NAME} .

if [ -n "${script_to_run}" ]; then

  echo "/* Running Docker Image */"

  docker run --rm ${INTERACTIVE} \
	 --gpus all \
	 -u $(id -u):$(id -g) \
	 ${NOTEBOOK_PATH:+ -v "$NOTEBOOK_PATH:/opt/disfluency_generator/notebooks"} \
         -v "${DATA_DIR}:/opt/disfluency_generator/data" \
	 ${PORT:+ -p "$PORT:$PORT"} \
         "${IMAGE_NAME}" ${script_to_run}
fi

