#!/usr/bin/env bash
set -e

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

SCRIPT_DIR=/workspace-SR006.nfs2/bulatov/rmt/test-time/test_time_gd
RUNS_DIR=/workspace-SR006.nfs2/bulatov/rmt/runs

NP=${NP:-1}  # Number of processes (default 1)
LR=3e-04
TBS=256
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

# Total tokens: 1 000 000 000
# n iters = 1 000 000 000 / 256 / (2 * 512) = 3814
MAX_STEPS=20000

# Model and dataset parameters
BASE_MODEL=gpt2
RMT_MODEL=rmt
SEGMENT_SIZE=512
MAX_N_SEGMENTS=2

N_CTRL_TOKENS=0
USE_MEM_PROJ=false
MEM_PROJ_MODE="none"
SAMPLE_SIZE=$SEGMENT_SIZE
DATASET_NAME=tt-fineweb-edu
TASK_NAME="HuggingFaceFW/fineweb-edu"
TOKENIZER_PATH=$BASE_MODEL  # or path to tokenizer if needed

# Output/experiment naming
for N_MEM_TOKENS in 4 16 32; do
  RUN_NAME=${BASE_MODEL}_mem${N_MEM_TOKENS}_max_n_segments${MAX_N_SEGMENTS}
  if [ "$N_CTRL_TOKENS" -gt 0 ]; then
    RUN_NAME=${RUN_NAME}_c${N_CTRL_TOKENS}
  fi
  if [ "$USE_MEM_PROJ" = true ]; then
    RUN_NAME=${RUN_NAME}_mem_proj
  fi
  RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

  N_VALUES=(1)
  for N in "${N_VALUES[@]}"; do
    EXP_PATH="${RUNS_DIR}/${DATASET_NAME}/${RUN_NAME}/run_$N"

    echo "RUNNING: DATASET_NAME $DATASET_NAME SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS"
    echo "SAMPLE_SIZE $SAMPLE_SIZE BASE_MODEL $BASE_MODEL LR $LR N $N"
    echo "gradient accumulation steps $GRAD_ACC_STEPS"

    accelerate launch \
      --main_process_port $((29500+$TBS+$N+1)) \
      --num_processes $NP \
      --mixed_precision bf16 \
      --config_file "${SCRIPT_DIR}/deepspeed_bf16.yaml" \
      "${SCRIPT_DIR}/run_original_rmt_on_lm.py" \
      --exp_path "$EXP_PATH" \
      --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACC_STEPS \
      --total_batch_size $TBS \
      --task_name "$TASK_NAME" \
      --tokenizer_path "$TOKENIZER_PATH" \
      --learning_rate $LR \
      --base_model $BASE_MODEL \
      --from_pretrained $BASE_MODEL \
      --rmt_model $RMT_MODEL \
      --n_mem_tokens $N_MEM_TOKENS \
      --n_ctrl_tokens $N_CTRL_TOKENS \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--use_mem_proj" ) \
      $( [ "$USE_MEM_PROJ" = true ] && echo "--mem_proj_mode $MEM_PROJ_MODE" ) \
      --max_n_segments $MAX_N_SEGMENTS \
      --segment_size $SEGMENT_SIZE \
      --sample_size $SAMPLE_SIZE \
      --val_sample_size $SAMPLE_SIZE \
      --max_steps $MAX_STEPS \
      --eval_steps 250 \
      --logging_steps 25 \
      --warmup_steps $(($MAX_STEPS/10)) \
      --early_stopping_patience 50 \
      --weight_decay 0.01 \
      --lr_scheduler_type constant_with_warmup \
      --metric_for_best_model "eval_loss" \
      --save_total_limit 2 \
      --bf16 \
      --seed $((42+$N))
  done
done

echo "Done"
