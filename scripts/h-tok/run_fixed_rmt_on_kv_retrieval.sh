#!/bin/bash
# HD-RMT Stage 1 — Ablation baseline: fixed-stride chunking on KV retrieval.
#
# Sweeps: PAIRS_PER_SEGMENT x N_MEM_TOKENS x LR
# N_PAIRS=16 is fixed so that we test genuine cross-segment recurrence and the
# comparison with the adaptive script (same N_PAIRS, same data) is apples-to-apples.
#
# Usage:
#   bash scripts/h-tok/run_fixed_rmt_on_kv_retrieval.sh
#   NP=4 bash scripts/h-tok/run_fixed_rmt_on_kv_retrieval.sh   # multi-GPU
set -e

NP=${NP:-1}
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(( TBS / (PER_DEVICE_BATCH_SIZE * NP) ))

# Model architecture (same as reference)
L=4
H=4
D=128
BASE_MODEL=llama

# KV task (same alphabet / data format as reference)
K=2
V=2
N_PAIRS=16
TOKENIZER_PATH="./tokenizers/kv_alphabet_62/"

# Fixed training budget
ITERS=200000

DATA_PATH_NAME="N${N_PAIRS}-K${K}V${V}-V62_1M"

for PAIRS_PER_SEGMENT in 1 2 4 8 16; do
  for N in 1 2; do
    for N_MEM_TOKENS in 4 8 32 64; do
      for LR in 1e-03 3e-04 5e-05; do

        RUN_NAME=fixed_rmt_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_N${N_PAIRS}_pps${PAIRS_PER_SEGMENT}_bs${TBS}_lr${LR}
        EXP_PATH="./runs-htok/${DATA_PATH_NAME}/${RUN_NAME}/run_${N}"

        if [ -d "$EXP_PATH" ]; then
          echo "Exists, skipping: $EXP_PATH"
          continue
        fi

        DATA_PATH="./data/${DATA_PATH_NAME}"

        accelerate launch \
          --main_process_port 0 \
          --num_processes $NP \
          --mixed_precision bf16 \
          --config_file accelerate.yaml \
          run_h_tok_on_kv_retrieval.py \
          --exp_path $EXP_PATH \
          --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
          --gradient_accumulation_steps $GRAD_ACC_STEPS \
          --total_batch_size $TBS \
          --data_path $DATA_PATH \
          --tokenizer_path $TOKENIZER_PATH \
          --base_model $BASE_MODEL \
          --n_layer $L \
          --n_head $H \
          --n_embd $D \
          --n_pairs $N_PAIRS \
          --n_keys $K \
          --n_values $V \
          --n_mem_tokens $N_MEM_TOKENS \
          --pairs_per_segment $PAIRS_PER_SEGMENT \
          --use_adaptive_chunking false \
          --learning_rate $LR \
          --max_steps $ITERS \
          --eval_steps 500 \
          --logging_steps 500 \
          --warmup_steps 10000 \
          --early_stopping_patience 500 \
          --seed $((142 + N))

      done
    done
  done
done

echo "Done: fixed RMT sweep"
