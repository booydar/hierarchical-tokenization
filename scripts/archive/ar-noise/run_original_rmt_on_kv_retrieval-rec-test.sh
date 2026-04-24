#!/bin/bash
set -e

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
TBS=256
PER_DEVICE_BATCH_SIZE=256
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
BASE_MODEL=llama
N_MEM_TOKENS=1

V=62

TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# RMT/GradMemGPT specific parameters (legacy, not used for RMT, but kept for compatibility)
N_CTRL_TOKENS=0
USE_MEM_PROJ=false
MEM_PROJ_MODE="proj"


N_SEGMENTS=1
PAIRS_PER_SEGMENT=8

MEMORY_TASK_FREQ=1
MEMORY_TASK="reconstruct"
LR=3e-04
for NOISE_LEVEL in 0.25 0.5 0.75; do
  DATA_PATH="N8-K2V2-V62_noise_${NOISE_LEVEL}_1M"
  RUN_NAME=original_rmt_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_lr${LR}-${N_SEGMENTS}x${PAIRS_PER_SEGMENT}_${MEMORY_TASK}_${MEMORY_TASK_FREQ}

    if [ "$N_CTRL_TOKENS" -gt 0 ]; then
      RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
    fi
    if [ "$USE_MEM_PROJ" = true ]; then
      RUN_NAME=${RUN_NAME}_mem_proj
    fi

    RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}-no_noise_rec

    # Path to save experiment results
  EXP_PATH="/workspace-SR006.nfs2/bulatov/rmt/runs/test-time-noise/${DATA_PATH}/${RUN_NAME}/run_$N"

    # Execute the script using accelerate for parallel processing
    accelerate launch \
    --main_process_port 0 \
      --num_processes $NP \
      --mixed_precision bf16 \
      --config_file accelerate.yaml \
      run_original_rmt_on_kv_retrieval-v3-noise-rec.py \
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
      --memory_task_freq $MEMORY_TASK_FREQ \
      --memory_task $MEMORY_TASK \
      --memory_key_size 4 \
      --memory_value_size 4 \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
      --pairs_per_segment $PAIRS_PER_SEGMENT \
      --max_steps 200000 \
      --eval_steps 500 \
      --logging_steps 500 \
      --warmup_steps 10000 \
      --early_stopping_patience 500 \
      --seed $((142 + N))
  done
echo "Done"
