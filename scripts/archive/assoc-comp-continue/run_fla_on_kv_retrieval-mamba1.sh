#!/bin/bash

NP=${NP:-1}
ADAM_BETA2=0.999
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(($TBS/($PER_DEVICE_BATCH_SIZE*$NP)))

H=1
K=2
N_VALUES=(1 2 3)
LR_VALUES=(1e-03)

# Dataset/tokenizer convention from the Mamba rebuttal sweeps.
VOCAB_SIZE=62
TOKENIZER_PATH="./tokenizers/kv_alphabet_${VOCAB_SIZE}/"

# Keep mamba2 if that is your intended FLA backend.
# If you want exact architectural parity with the original Mamba sweeps, change this to: BASE_MODELS=(mamba)
BASE_MODELS=(mamba)

# Union of:
# - run_mamba_on_kv_retrieval-all-conf-more.sh
# - run_mamba_on_kv_retrieval-all-conf-more-2.sh
# - run_mamba_on_kv_retrieval-all-conf-more-3.sh
# - run_mamba_on_kv_retrieval-all-conf-more-4.sh
# Format: D:L:STATE_SIZE:CONV_KERNEL
SWEEP_CONFIGS=(
  "64:4:16:4"
  "64:1:4:4"
  "64:2:2:2"
  "64:4:1:1"
  "64:4:4:4"

  "256:4:16:4"
  "256:1:4:4"
  "256:2:2:2"
  "256:4:1:1"
  "256:4:4:4"

  "128:2:4:4"
  "128:2:16:4"
  "128:2:2:4"
  "128:1:4:4"
  "128:1:16:4"
  "128:1:2:4"

  "128:2:4:2"
  "128:2:16:2"
  "128:2:2:2"
  "128:1:4:2"
  "128:1:16:2"
  "128:1:2:2"

  "128:2:32:2"
  "128:2:32:4"
  "64:2:32:2"
  "64:2:32:4"
  "32:2:32:2"
  "32:2:32:4"
)

for BASE_MODEL in "${BASE_MODELS[@]}"; do
  for N_PAIRS in 8 16 32; do
    DATA_NAME="N${N_PAIRS}-K${K}V${K}-V${VOCAB_SIZE}_1M"
    DATA_PATH="./data/ar/${DATA_NAME}"
    # If your FLA datasets are still under ./data/ instead of ./data/ar/, use:
    # DATA_PATH="./data/${DATA_NAME}"

    for N in "${N_VALUES[@]}"; do
      for LR in "${LR_VALUES[@]}"; do
        for CFG in "${SWEEP_CONFIGS[@]}"; do
          IFS=':' read -r D L STATE_SIZE CONV_KERNEL <<< "$CFG"

          RUN_NAME="${BASE_MODEL}_L${L}D${D}_ss${STATE_SIZE}_ck${CONV_KERNEL}"
          RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

          if [ -n "$ADAM_BETA2" ]; then
            RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
          fi

          EXP_PATH="./runs-rebuttal/${DATA_NAME}/${RUN_NAME}/run_${N}"

          accelerate launch \
            --main_process_port $((29500 + TBS + N + 1)) \
            --num_processes $NP \
            --mixed_precision bf16 \
            --config_file accelerate.yaml \
            run_fla_on_kv_retrieval-2.py \
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
            --state_size $STATE_SIZE \
            --conv_kernel $CONV_KERNEL \
            --n_pairs $N_PAIRS \
            --n_keys $K \
            --n_values $K \
            --base_model $BASE_MODEL \
            --max_steps 200000 \
            --eval_steps 500 \
            --logging_steps 500 \
            --warmup_steps 10000 \
            --early_stopping_patience 500 \
            --seed $((142 + N))
        done
      done
    done
  done
done

echo "Done"