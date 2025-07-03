#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/metric-init.sh

set -euxo pipefail

apk add git --no-cache

if (( PARAM_IQA_PYTORCH != 0 )); then
    git submodule update --init --recursive --depth=1 subjects/IQA-PyTorch
fi
git submodule sync
git submodule update --init --recursive --depth=1 "subjects/$METRIC_NAME"
cd "subjects/$METRIC_NAME"

if (( PARAM_IQA_PYTORCH != 0 )); then
  docker build -t "$IMAGE" -f Dockerfile ..
else
  docker build -t "$IMAGE" "$PARAM_DOCKER_BUILD_PATH"
fi
docker push "$IMAGE"
