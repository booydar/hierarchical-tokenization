#!/bin/bash

# Define arguments for the script
NP=${NP:-1}  # Default to 1 process if not set
ADAM_BETA2=0.99
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

L=4
H=1
D=128
MAX_POSITION_EMBEDDINGS=1024
# FLASHRNN_BACKEND=${FLASHRNN_BACKEND:-vanilla}  # vanilla, cuda, cuda_fused, triton_fused
FLASHRNN_BACKEND=${FLASHRNN_BACKEND:-cuda_fused}  # vanilla, cuda, cuda_fused, triton_fused

K=2
V=2
VOCAB=62
# Dataset parameters
TOKENIZER_PATH="./tokenizers/kv_alphabet_${VOCAB}/"

# for BASE_MODEL in flashrnn_lstm flashrnn_gru flashrnn_elman flashrnn_slstm; do
for N_PAIRS in 1 2 4; do
  for BASE_MODEL in flashrnn_gru flashrnn_elman flashrnn_lstm flashrnn_slstm; do
    for LR in 1e-03 3e-04;
    do
      # N_PAIRS=$((N_SEGMENTS * PAIRS_PER_SEGMENT))
      DATA_NAME="N${N_PAIRS}-K${K}V${V}-V62_1M"
      DATA_PATH="./data/${DATA_NAME}"
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
          --n_pairs $N_PAIRS \
          --n_keys $K \
          --n_values $V \
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
done
echo "Done"
