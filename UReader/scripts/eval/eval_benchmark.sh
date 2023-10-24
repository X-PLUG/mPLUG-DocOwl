export PYTHONPATH=`pwd`
python -m torch.distributed.launch --use_env \
    --nproc_per_node ${NPROC_PER_NODE:-8} \
    --nnodes ${WORLD_SIZE:-1} \
    --node_rank ${RANK:-0} \
    --master_addr ${MASTER_ADDR:-127.0.0.1} \
    --master_port ${MASTER_PORT:-12345} \
    pipeline/evaluation.py \
    --hf_model ./checkpoints/ureader