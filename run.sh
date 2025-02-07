#!bin/bash

run() {

    local DISABLE_SCATTERMOE=${1:-"False"}
    local N_EXPERT=${2:-2}
    
    DISABLE_SCATTERMOE=$DISABLE_SCATTERMOE \
    torchrun --nnodes=1 --node_rank=0 \
        --nproc_per_node=$N_EXPERT --rdzv_id=101 \
        --rdzv_endpoint="localhost:27501" \
        -m train \
        --model_name /home/flim/data/Bamba-9b-2.3T \
        --per_device_train_batch_size 8 \
        --low_cpu_mem_mode true \
        --n_expert $N_EXPERT \
        --max_seq_length 128
}

LOG_DIR=exprs

mkdir -p $LOG_DIR

TAG="kernels"
for disable in "False" "True" ; do
    for ne in 2 4 8 ; do
        NAME=$LOG_DIR/${TAG}_${disable}_${ne}
        run $disable $ne 1> $NAME.log 2> $NAME.err
    done
done