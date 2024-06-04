#!/bin/bash
TRAIN_DATA=data/train.json
TEST_DATA=data/test.json

LLM_PATH=bczhou/TinyLLaVA-3.1B
VIT_PATH=pretrained_models/TinyLLaVA-3.1B-SigLIP

# # If you want to fine-tune TinyChart-3B-768:
# LLM_PATH=mPLUG/TinyChart-3B-768
# VIT_PATH=mPLUG/TinyChart-3B-768-siglip

OUTPUT=./checkpoints/TinyChart-3B

mkdir -p ${OUTPUT}
# Copy the script to OUTPUT directory
cp scripts/train.sh ${OUTPUT}/

export PYTHONPATH=./

if [ $MASTER_ADDR ];then
	echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
fi
# Change for multinode config
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS \
    tinychart/train/train.py \
    --lora_enable False \
    --tune_vision_tower True \
    --tune_entire_model True \
    --tune_vit_from_layer -1 \
    --deepspeed scripts/zero3_offload_decay.json \
    --model_name_or_path ${LLM_PATH} \
    --vision_tower ${VIT_PATH} \
    --version v1 \
    --data_path ${TRAIN_DATA} \
    --image_folder '' \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --bf16 False \
    --output_dir ${OUTPUT} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 10 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
2>&1 | tee -a ${OUTPUT}/log.${RANK}.txt

# Evaluate
if [ $RANK -eq 0 ]; then
    python scripts/convert_model_config.py --input ${OUTPUT}
    bash scripts/evaluate.sh ${OUTPUT} ${TEST_DATA}
fi