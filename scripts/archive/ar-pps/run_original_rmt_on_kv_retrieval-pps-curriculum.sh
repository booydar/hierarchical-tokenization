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

N=1
LR=3e-04

V=62

TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# RMT/GradMemGPT specific parameters (legacy, not used for RMT, but kept for compatibility)
N_CTRL_TOKENS=0
USE_MEM_PROJ=false
MEM_PROJ_MODE="proj"


N_SEGMENTS_VALUES=(1 1 2 2 4 4)
PAIRS_PER_SEGMENT_VALUES=(4 8 8 16 16 32)


for i in "${!PAIRS_PER_SEGMENT_VALUES[@]}"; do
  PAIRS_PER_SEGMENT=${PAIRS_PER_SEGMENT_VALUES[$i]}
  N_SEGMENTS=${N_SEGMENTS_VALUES[$i]}

  N_PAIRS=$((N_SEGMENTS * PAIRS_PER_SEGMENT))
  DATA_PATH="N${N_PAIRS}-K2V2-V62_1M"

  RUN_NAME=original_rmt_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_lr${LR}-${N_SEGMENTS}x${PAIRS_PER_SEGMENT}

  if [ $i -eq 0 ]; then
    PREV_RUN_NAME=None
    PREV_EXP_PATH=None
  else
    PREV_PAIRS_PER_SEGMENT=${PAIRS_PER_SEGMENT_VALUES[$((i-1))]}
    PREV_N_SEGMENTS=${N_SEGMENTS_VALUES[$((i-1))]}
    
    PREV_N_PAIRS=$((PREV_N_SEGMENTS * PREV_PAIRS_PER_SEGMENT))
    PREV_DATA_PATH="N${PREV_N_PAIRS}-K2V2-V62_1M"
    # RUN_NAME=${RUN_NAME}-from${PREV_N_SEGMENTS}x${PREV_PAIRS_PER_SEGMENT}
    PREV_RUN_NAME=original_rmt_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_lr${LR}-${PREV_N_SEGMENTS}x${PREV_PAIRS_PER_SEGMENT}
    if [ "$N_CTRL_TOKENS" -gt 0 ]; then
      PREV_RUN_NAME=${PREV_RUN_NAME}_c${N_CTRL_TOKENS}
    fi
    if [ "$USE_MEM_PROJ" = true ]; then
      PREV_RUN_NAME=${PREV_RUN_NAME}_mem_proj
    fi

    PREV_RUN_NAME=${PREV_RUN_NAME}_bs_${TBS}_lr_${LR}
    PREV_EXP_PATH="/workspace-SR006.nfs2/bulatov/rmt/runs/test-time-pps-v1/${PREV_DATA_PATH}/${PREV_RUN_NAME}/run_$N"
  fi

  if [ "$N_CTRL_TOKENS" -gt 0 ]; then
    RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
  fi
  if [ "$USE_MEM_PROJ" = true ]; then
    RUN_NAME=${RUN_NAME}_mem_proj
  fi

  RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

  # Path to save experiment results
  EXP_PATH="/workspace-SR006.nfs2/bulatov/rmt/runs/test-time-pps-v1/${DATA_PATH}/${RUN_NAME}/run_$N"
  # if path exists, skip
  if [ -d "$EXP_PATH" ]; then
    echo "Path $EXP_PATH already exists, skipping"
    continue
  fi

  # Execute the script using accelerate for parallel processing
  accelerate launch \
    --main_process_port 0 \
    --num_processes $NP \
    --mixed_precision bf16 \
    --config_file accelerate.yaml \
    run_original_rmt_on_kv_retrieval-v3.py \
    --exp_path $EXP_PATH \
    --model_cpt $PREV_EXP_PATH \
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
    --pairs_per_segment $PAIRS_PER_SEGMENT \
    --max_steps 200000 \
    --eval_steps 500 \
    --logging_steps 500 \
    --warmup_steps 10000 \
    --early_stopping_patience 500 \
    --seed $((142 + N))
done

echo "Done"
