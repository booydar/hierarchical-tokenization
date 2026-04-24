#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
ADAM_BETA2=0.99
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=4
D=128
MAX_POSITION_EMBEDDINGS=1024
# FLASHRNN_BACKEND=${FLASHRNN_BACKEND:-vanilla}  # vanilla, cuda, cuda_fused, triton_fused
FLASHRNN_BACKEND=${FLASHRNN_BACKEND:-cuda_fused}  # vanilla, cuda, cuda_fused, triton_fused

V=62
# Dataset parameters
DATA_NAME="N8-K2V2-V${V}_1M"
DATA_PATH="./data/${DATA_NAME}"
TOKENIZER_PATH="./tokenizers/kv_alphabet_${V}/"

# for BASE_MODEL in flashrnn_lstm flashrnn_gru flashrnn_elman flashrnn_slstm; do
LR=1e-03
for BASE_MODEL in flashrnn_slstm; do
  for L in 1 8 16;
  do

    RUN_NAME="${BASE_MODEL}_L${L}H${H}D${D}"
    RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

    if [ -n "$ADAM_BETA2" ]; then
      RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
    fi

    if [ -n "$RUN_NAME_SUFFIX" ]; then
      RUN_NAME=${RUN_NAME}_${RUN_NAME_SUFFIX}
    fi


    N_VALUES=(1)
    for N in "${N_VALUES[@]}"; do
      EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_$N"

      accelerate launch \
        --main_process_port $((29500+$TBS+$N+1)) \
        --num_processes $NP \
        --mixed_precision no \
        --config_file accelerate.yaml \
        run_lstm_on_kv_retrieval.py \
        --exp_path $EXP_PATH \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACC_STEPS \
        --total_batch_size $TBS \
        --data_path $DATA_PATH \
        --tokenizer_path $TOKENIZER_PATH \
        --learning_rate $LR \
        $( [ -n "$ADAM_BETA2" ] && echo "--adam_beta2 $ADAM_BETA2" ) \
        --n_layer $L \
        --n_head $H \
        --n_embd $D \
        --base_model $BASE_MODEL \
        --flashrnn_backend $FLASHRNN_BACKEND \
        --max_steps 200000 \
        --eval_steps 500 \
        --logging_steps 500 \
        --warmup_steps 10000 \
        --early_stopping_patience 500 \
        --seed $((142+$N))
    done
  done
done
echo "Done"
