#!/bin/bash
set -e

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=1
D=128
BASE_MODEL=llama
N_MEM_TOKENS=8

LR=3e-04

K=2
V=2

TOKENIZER_PATH="./tokenizers/kv_alphabet_62/"

# RMT/GradMemGPT specific parameters (legacy, not used for RMT, but kept for compatibility)
N_CTRL_TOKENS=0
USE_MEM_PROJ=false
MEM_PROJ_MODE="proj"

N_SEGMENTS=1
N_MEM_TOKENS=8

for PAIRS_PER_SEGMENT in 1; do
  for LR in 3e-04; do
    for N in 1; do

      N_PAIRS=$((N_SEGMENTS * PAIRS_PER_SEGMENT))
      DATA_PATH="N${N_PAIRS}-K${K}V${V}-V62_1M"

      RUN_NAME=rmca_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_lr${LR}-${N_SEGMENTS}x${PAIRS_PER_SEGMENT}

      if [ "$N_CTRL_TOKENS" -gt 0 ]; then
        RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
      fi
      if [ "$USE_MEM_PROJ" = true ]; then
        RUN_NAME=${RUN_NAME}_mem_proj
      fi

      RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}-gen

      # Path to save experiment results
      EXP_PATH="./runs/${DATA_PATH}/${RUN_NAME}/run_$N"
      # if path exists, skip
      if [ -d "$EXP_PATH" ]; then
        echo "Path $EXP_PATH already exists, skipping"
        continue
      fi
      DATA_PATH="./data/${DATA_PATH}"

      # Execute the script using accelerate for parallel processing
      accelerate launch \
        --main_process_port 0 \
        --num_processes $NP \
        --mixed_precision bf16 \
        --config_file accelerate.yaml \
        run_rmca_on_kv_retrieval.py \
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
        --n_pairs $N_PAIRS \
        --n_keys $K \
        --n_values $V \
        --base_model $BASE_MODEL \
        --n_mem_tokens $N_MEM_TOKENS \
        --n_ctrl_tokens $N_CTRL_TOKENS \
        $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
        $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
        --pairs_per_segment $PAIRS_PER_SEGMENT \
        --max_steps 100000 \
        --eval_steps 500 \
        --logging_steps 500 \
        --warmup_steps 10000 \
        --early_stopping_patience 500 \
        --seed $((142 + N))
    done
  done
done

echo "Done"
