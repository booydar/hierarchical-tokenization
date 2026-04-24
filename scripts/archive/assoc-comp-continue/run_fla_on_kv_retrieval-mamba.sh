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
BASE_MODELS=(mamba2)

# Union of:
# - run_mamba_on_kv_retrieval-all-conf-more.sh
# - run_mamba_on_kv_retrieval-all-conf-more-2.sh
# - run_mamba_on_kv_retrieval-all-conf-more-3.sh
# - run_mamba_on_kv_retrieval-all-conf-more-4.sh
# Format: D:L:STATE_SIZE:CONV_KERNEL

N_LAYER_VALUES=(1 2 4)
N_EMBD_VALUES=(64 128 256)
STATE_SIZE_VALUES=(4 16 32)
CONV_KERNEL_VALUES=(2 4)

for BASE_MODEL in "${BASE_MODELS[@]}"; do
  for N_PAIRS in 8 16 32; do
    for N_LAYER in "${N_LAYER_VALUES[@]}"; do
      for N_EMBD in "${N_EMBD_VALUES[@]}"; do
        for STATE_SIZE in "${STATE_SIZE_VALUES[@]}"; do
          for CONV_KERNEL in "${CONV_KERNEL_VALUES[@]}"; do
            DATA_NAME="N${N_PAIRS}-K${K}V${K}-V${VOCAB_SIZE}_1M"
            DATA_PATH="./data/ar/${DATA_NAME}"
            # If your FLA datasets are still under ./data/ instead of ./data/ar/, use:
            # DATA_PATH="./data/${DATA_NAME}"

            for N in "${N_VALUES[@]}"; do
              for LR in "${LR_VALUES[@]}"; do
                  RUN_NAME="${BASE_MODEL}_L${N_LAYER}D${N_EMBD}_ss${STATE_SIZE}_ck${CONV_KERNEL}"
                  RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

                  if [ -n "$ADAM_BETA2" ]; then
                    RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
                  fi

                  EXP_PATH="./runs-rebuttal/${DATA_NAME}/${RUN_NAME}/run_${N}"
                  if [ -d "$EXP_PATH" ]; then
                    echo "Experiment path already exists: $EXP_PATH"
                    continue
                  fi
             
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
                    --n_layer $N_LAYER \
                    --n_head $H \
                    --n_embd $N_EMBD \
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
        done
      done
    done
  done
done

echo "Done"