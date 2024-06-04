#!/bin/bash
# Variables
MODEL_PATH=$1
TEST_DATA_PATH=$2

OUTPUT=${MODEL_PATH}/eval
mkdir -p ${OUTPUT}
cp scripts/evaluate.sh ${OUTPUT}/


export PYTHONPATH=./
export PYTHONHASHSEED=42
export PYTHONUNBUFFERED=1

num_chunks=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
time_stamp=$(date +%Y%m%d-%H%M%S)
TEMP_DIR=${OUTPUT}/temp_${time_stamp}
mkdir -p ${TEMP_DIR}

for ((chunk_idx=0; chunk_idx<num_chunks; chunk_idx++)); do
    CUDA_VISIBLE_DEVICES=$chunk_idx python -u tinychart/eval/eval_model.py \
        --model_path ${MODEL_PATH} \
        --data_path ${TEST_DATA_PATH} \
        --image_folder '' \
        --output_path ${TEMP_DIR}/evaluate.${chunk_idx}.jsonl \
        --num_chunks ${num_chunks} \
        --chunk_idx ${chunk_idx} \
        --max_new_tokens 1024 2>&1 | tee -a ${OUTPUT}/log.txt &
done
wait

# Merge split && divide by dataset && calculate metric
python scripts/merge_jsonl_sort.py \
    --input ${TEMP_DIR} \
    --output ${TEMP_DIR}/all.jsonl
python scripts/split_jsonl_dataset.py \
    --input ${TEMP_DIR}/all.jsonl \
    --output ${OUTPUT}
python tinychart/eval/run_eval.py \
    --input ${OUTPUT}
