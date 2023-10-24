#!/bin/bash
# For V100 32G
DIR=`pwd`
export PYTHONPATH=$DIR
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

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

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

EXP_NAME=ureader

max_length=2048
micro_batch_size=1
global_batch_size=256
gradient_accumulation_steps=1

SAVE_NAME=${ureader}_${max_length}_${global_batch_size}

SAVE_PATH="./output/${EXP_NAME}/"
TENSORBOARD_PATH="./tensorboard/sft/${SAVE_NAME}/"



# train_iters = total_data * train_epochs // global_batch_size
train_epochs=10
train_iters=25579

lr_warmup_iters=50

eval_iter=50
eval_interval=50
save_interval=500

mkdir -p ${SAVE_PATH}
mkdir -p ${TENSORBOARD_PATH}

options=" \
	--pretrained-ckpt checkpoints/mplug-owl-llama-7b \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
	--global-batch-size ${global_batch_size} \
	--num-training-steps ${train_iters} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr 1e-4 \
	--min-lr 1e-6 \
	--eval-iters ${eval_iter} \
    --save-interval ${save_interval} \
	--save-path ${SAVE_PATH} \
	--tensorboard-dir ${TENSORBOARD_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 16 \
	--use-lora \
	--gradient-checkpointing \
	--fp16 \
	--deepspeed ds_config.json"

multimodal_options=" \
	--mm-config configs/sft/release.yaml
    "

python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pipeline/train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 