#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e

SCRIPT_DIR=/workspace-SR006.nfs2/bulatov/rmt/test-time/test_time_gd
RUNS_DIR=/workspace-SR006.nfs2/bulatov/rmt/runs


CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
DATASET_NAME=fineweb-edu
METRIC=exact_match

# MODEL_NAME=SmolLM2-360M
# FROM_PRETRAINED=HuggingFaceTB/SmolLM2-135M
MODEL_NAME=gpt2
FROM_PRETRAINED=gpt2

RMT_MODEL=rmt

TASK_DATASET=HuggingFaceFW/fineweb-edu


LR=3e-04
SEGMENT_SIZE=1024

# First iteration
MAX_N_SEGMENTS=2
SAMPLE_SIZE=$SEGMENT_SIZE # length of task sample in tokens
SCHEDULER=constant

TBS=256
echo TBS $TBS
BS=64
GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))

ITERS=5000

N=1

K2=-1   # BPTT unroll length

ACCEL_CONFIG="${SCRIPT_DIR}/accel_configs/deepspeed_bf16.yaml"
MAIN_SCRIPT="${SCRIPT_DIR}/run_original_rmt_on_lm.py"

echo RUNNING: DATASET_NAME $DATASET_NAME MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --num_processes $NP --config_file $ACCEL_CONFIG --main_process_port 29030 $MAIN_SCRIPT \
        --task_name $TASK_DATASET \
        --output_dir ${RUNS_DIR}/${DATASET_NAME}/$MODEL_NAME/LR${LR}_${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}-nfs/run_$N \
        --model_type $MODEL_TYPE \
        --model_cls $BACKBONE_CLS \
        --from_pretrained $FROM_PRETRAINED \
        --rmt_model $RMT_MODEL \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --val_sample_size $SAMPLE_SIZE \
        --per_device_train_batch_size $BS --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --max_steps $ITERS \
        --metric_for_best_model "eval_loss" \
        --greater_is_better false \
        --save_total_limit 1 \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --learning_rate ${LR} --lr_scheduler_type $SCHEDULER --warmup_steps 3000 \
        --data_n_workers 2 \
        --logging_steps 25 --eval_steps 100 \
        --show_valid_examples 5 \
        --seed $(($N+42)) \
        --report_to tensorboard \


echo "done"
