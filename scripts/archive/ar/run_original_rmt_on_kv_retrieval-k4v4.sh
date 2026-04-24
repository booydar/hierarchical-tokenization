#!/bin/bash
set -e

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
BASE_MODEL=llama
N_MEM_TOKENS=8

V=62
# Dataset parameters
# DATA_NAME="N2-K4V4-S4(32-64)_1M"
# DATA_NAME="N2-K4V4-S1(16-32)_1M"
# DATA_NAME="N2-K4V4-S2(16-32)_1M"
# DATA_NAME="N0-S1(4-4)_1M"
# DATA_NAME="N10-K2V2-S4(32-64)_1M"
# DATA_NAME="N8-K1V1-vocab512-no_noise_1M"
# DATA_NAME="N4-K2V2-V${V}_1M"
# DATA_PATH="./data/${DATA_NAME}"
DATA_PATH="N8-K4V4-V62_1M"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# RMT/GradMemGPT specific parameters (legacy, not used for RMT, but kept for compatibility)
N_CTRL_TOKENS=0
USE_MEM_PROJ=false
MEM_PROJ_MODE="proj"

LR=1e-04

for H in 2 4; do
  for N in 1 2; do
    RUN_NAME=original_rmt_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_lr${LR}
    if [ "$N_CTRL_TOKENS" -gt 0 ]; then
      RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
    fi
    if [ "$USE_MEM_PROJ" = true ]; then
      RUN_NAME=${RUN_NAME}_mem_proj
    fi

    RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

    # Path to save experiment results
    EXP_PATH="/workspace-SR006.nfs2/bulatov/rmt/runs/test-time/${DATA_PATH}/${RUN_NAME}/run_$N"

    # Execute the script using accelerate for parallel processing
    accelerate launch \
      --main_process_port $((29500+$TBS+$N+1)) \
      --num_processes $NP \
      --mixed_precision bf16 \
      --config_file accelerate.yaml \
      run_original_rmt_on_kv_retrieval.py \
      --exp_path $EXP_PATH \
      --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACC_STEPS \
      --total_batch_size $TBS \
      --data_path $DATA_PATH \
      --tokenizer_path $TOKENIZER_PATH \
      --learning_rate $LR \
      --n_layer $L \
      --n_head $H \
      --n_embd $D \
      --base_model $BASE_MODEL \
      --n_mem_tokens $N_MEM_TOKENS \
      --n_ctrl_tokens $N_CTRL_TOKENS \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
      --max_steps 200000 \
      --eval_steps 500 \
      --logging_steps 500 \
      --warmup_steps 10000 \
      --early_stopping_patience 500 \
      --seed $((142 + N))
  done
done

echo "Done"
