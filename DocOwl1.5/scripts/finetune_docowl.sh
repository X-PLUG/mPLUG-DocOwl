#!/bin/bash
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
# GPUS_PER_NODE=1
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
echo $DISTRIBUTED_ARGS

# change LOAD to your local path of DocOwl1.5-stage1
LOAD='./mPLUG/DocOwl1.5-stage1'

# batch size = per_device_train_batch_size x GPUS_PER_NODE x NNODES x gradient_accumulation_steps
DATA_FILE=./DocDownstream-1.0/train.jsonl
torchrun $DISTRIBUTED_ARGS mplug_docowl/train/train_docowl.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --data_path $DATA_FILE \
    --image_folder './DocDownstream-1.0/' \
    --image_size 448 \
    --crop_anchors 'grid_9' \
    --add_global_img True \
    --add_textual_crop_indicator True \
    --bf16 True \
    --output_dir ./checkpoints/docowl1.5 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 4 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3600 \
    --gradient_checkpointing True \
    --tune_vision2text True \
    --freeze_vision_model True \
    --freeze_backbone False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard