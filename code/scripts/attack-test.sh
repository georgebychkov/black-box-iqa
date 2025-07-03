#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/attack-init.sh
. "$CI_PROJECT_DIR"/scripts/attack-generate-pipeline.sh

set -euxo pipefail
shopt -s extglob

trap
trap 'echo TRAPPED! "$@"' err

DATASETS_STORAGE="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/"


load_method_trainable_blackbox "$METHOD_NAME"
load_method_non_uap_blackbox "$METHOD_NAME"

if [[ "$METHOD_TRAINABLE_BLACKBOX" == 1 ]]; then
 
	TRAIN_DATASETS=( BLACK-BOX )
	TRAIN_DATASET_PATHS=( 
		"/train/black-box-dataset"
	)
else
	TRAIN_DATASETS=( COCO VOC2012 )
	TRAIN_DATASET_PATHS=( 
		"/train/COCO_25e_shu/train"
		"/train/VOC2012/JPEGImages"
	)
fi

if [[ "$NON_UAP_BLACKBOX_METHODS" == 1 ]]; then
  TEST_DATASETS=( SMALL-KONIQ-50 )
	TEST_DATASET_PATHS=(
    "/test/quality-sampled-datasets/koniq_sampled_MOS/50_10_clusters"
	)
else
	TEST_DATASETS=( SMALL-KONIQ-50)
  TEST_DATASET_PATHS=( 
  #  "/test/DERF"
    "/test/quality-sampled-datasets/koniq_sampled_MOS/50_10_clusters"
  #  "/test/vimeo_triplet_images/test"
  #  "/test/NIPS 2017/images"
  )
fi
 

load_metric_launch_params "$METRIC_NAME" "$METHOD_NAME"
load_method_trainable "$METHOD_NAME"
load_method_multimetric "$METHOD_NAME"
load_video_metric "$METRIC_NAME"

if [[ "$VIDEO_METRIC" == 1 ]]; then
    video_param="--video-metric"
#    TEST_DATASETS=( DERF )
#    TEST_DATASET_PATHS=( 
#        "/test/DERF"
#     )
else
    video_param=""
fi





is_fr=($(jq -r '.is_fr' "${CI_PROJECT_DIR}/subjects/${METRIC_NAME}/config.json"))

if [[ "$is_fr" == true ]]; then
    quality_param="--jpeg-quality 80"
else
    quality_param=""
fi

if [[ "$METHOD_NAME" == "noattack" ]]; then
    if [[ "$is_fr" == true ]]; then
        TEST_DATASETS=( NIPS_200 )
        TEST_DATASET_PATHS=( 
            "/test/NIPS 2017/images_200"
         )
    else
        TEST_DATASETS=( DIV2K_valid_HR )
        TEST_DATASET_PATHS=( 
            "/test/DIV2K_valid_HR"
         )
    fi
fi



codecs_param="--codec libx264 libx265"



cd "methods/$METHOD_NAME"
cp -a run.py "$CACHE/"


DUMPS_STORAGE="${CACHE}/dumps"
mkdir -p DUMPS_STORAGE


DOCKER_PARAMS=( --init --gpus device="${CUDA_VISIBLE_DEVICES-0}" -t --rm --name "gitlab-$CI_PROJECT_PATH_SLUG-$CI_JOB_ID" )

if (( PARAM_TRAIN != 0 )); then

    cp -a train.py "$CACHE/"


    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/train":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/train.py:/train.py" \
      "$IMAGE" \
      python ./train.py \
        "${METRIC_LAUNCH_PARAMS[@]}" \
        --path-train "${TRAIN_DATASET_PATHS[@]}" \
        --metric  "${METRIC_NAME}" \
        --train-dataset "${TRAIN_DATASETS[@]}" \
        --save-dir /artifacts \
        $quality_param \
        --device "cuda:0" \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.log"
    
    for train_dataset in "${TRAIN_DATASETS[@]}"; do
            mv "$CACHE/${train_dataset}.npy" "$CI_PROJECT_DIR/${METRIC_NAME}_${train_dataset}.npy"
            mv "$CACHE/${train_dataset}.png" "$CI_PROJECT_DIR/${METRIC_NAME}_${train_dataset}.png"
    done


elif (( METHOD_TRAINABLE != 0 )); then
    
    mkdir -p "$CACHE/uap"
    UAP_PATHS=()
    for train_dataset in "${TRAIN_DATASETS[@]}"; do
        uap_fn="${METRIC_NAME}_${train_dataset}.npy"
        UAP_PATHS+=("/uap/${uap_fn}")
        cp -a "$CI_PROJECT_DIR/${uap_fn}" "$CACHE/uap"
    done
    
    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/test":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/run.py:/run.py" \
      -v "$CACHE/uap:/uap" \
      -v "$DUMPS_STORAGE":"/dumps" \
      "$IMAGE" \
      python ./run.py \
        "${METRIC_LAUNCH_PARAMS[@]}" \
        --amplitude 0.2 0.4 0.6 0.8 1.0 2.0 4.0 8.0\
        --metric  "${METRIC_NAME}" \
        --uap-path "${UAP_PATHS[@]}" \
        --train-dataset "${TRAIN_DATASETS[@]}" \
        --test-dataset "${TEST_DATASETS[@]}" \
        --dataset-path "${TEST_DATASET_PATHS[@]}" \
        $quality_param \
        $video_param \
        $codecs_param \
        --save-path "/artifacts/${METRIC_NAME}_test.csv" \
        --device "cuda:0" \
        --dump-path "/dumps" \
        --dump-freq 1 \
        --log-file "/artifacts/log.csv" \
        --job-id $CI_JOB_ID \
        --job-name $CI_JOB_NAME \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
    
    #mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"
    
    zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
    mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
    #mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
    cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"

    
  
elif (( METHOD_MULTIMETRIC != 0 )); then 
    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/test":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/run.py:/run.py" \
      -v "$DUMPS_STORAGE":"/dumps" \
      "$IMAGE" \
      python ./run.py \
        --test-dataset "${TEST_DATASETS[@]}" \
        --metric  "${METRIC_NAME}" \
        --dataset-path "${TEST_DATASET_PATHS[@]}" \
        --metric-list "${METRICS[@]}" \
        --target-metric ${METRIC_NAME} \
        $quality_param \
        $video_param \
        $codecs_param \
        --save-path "/artifacts/${METRIC_NAME}_test.csv" \
        --device "cuda:0" \
        --dump-path "/dumps" \
        --dump-freq 1 \
        --log-file "/artifacts/log.csv" \
        --job-id $CI_JOB_ID \
        --job-name $CI_JOB_NAME \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
      
    #mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"
    
    zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
    mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
    #mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
    cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"
else

    
    
    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/test":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/run.py:/run.py" \
      -v "$DUMPS_STORAGE":"/dumps" \
      "$IMAGE" \
      python ./run.py \
        "${METRIC_LAUNCH_PARAMS[@]}" \
        --test-dataset "${TEST_DATASETS[@]}" \
        --metric  "${METRIC_NAME}" \
        --dataset-path "${TEST_DATASET_PATHS[@]}" \
        $quality_param \
        $video_param \
        $codecs_param \
        --save-path "/artifacts/${METRIC_NAME}_test.csv" \
        --device "cuda:0" \
        --dump-path "/dumps" \
        --dump-freq 1 \
        --log-file "/artifacts/log.csv" \
        --job-id $CI_JOB_ID \
        --job-name $CI_JOB_NAME \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
      
    #mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"
    
    zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
    mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
    #mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
    cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"


fi
