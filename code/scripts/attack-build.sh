#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/attack-init.sh

set -euxo pipefail

cd "methods/$METHOD_NAME"
load_method_multimetric "$METHOD_NAME"

load_cadv "$METHOD_NAME"

load_method_trainable "$METHOD_NAME"


cp -a "$CI_PROJECT_DIR"/methods/utils/. ./
cp -a "${GML_SHARED}/vqmt/." ./

if (( METHOD_TRAINABLE != 0 )); then
    printf "\nCOPY train.py /train.py\n" >> Dockerfile
fi

if (( METHOD_CADV != 0)); then
    printf "\nRUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/ydzYpLFwY/cadv-colorization-model.pth  https://titan.gml-team.ru:5003/fsdownload/ydzYpLFwY/cadv-colorization-model.pth \
 && rm cadv-colorization-model.pth.1\n" >> Dockerfile
fi


if (( METHOD_MULTIMETRIC != 0 )); then 
    for i in "${METRICS[@]}"
    do
    	printf "\nCOPY --from=${NEW_CI_REGISTRY}/metric/${i}:${LAUNCH_ID} /src /${i}/\n" >> Dockerfile
    	weights=($(jq -r '.weight' "${CI_PROJECT_DIR}/subjects/${i}/config.json"  | tr -d '[]," '))
    	for fn in "${weights[@]}"
        do
            printf "\nCOPY --from=${NEW_CI_REGISTRY}/metric/${i}:${LAUNCH_ID} /${fn} /${i}/${fn}\n" >> Dockerfile
        done
    done
    printf "\nCOPY run.py /run.py\n" >> Dockerfile
    docker build -t "$IMAGE" .
    
else
    docker build -t "$IMAGE" --build-arg METRIC_IMAGE="$NEW_CI_REGISTRY/metric/$METRIC_NAME:$LAUNCH_ID" .
fi
docker push "$IMAGE"
