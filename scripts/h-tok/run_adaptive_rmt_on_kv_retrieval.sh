#!/bin/bash
# HD-RMT Stage 1 — Adaptive chunking on KV retrieval.
#
# DynamicChunker replaces fixed-stride segmentation.
# N_CHUNKS matches PAIRS_PER_SEGMENT values in the fixed baseline so that
# both conditions have the same number of recurrent steps over the same data.
#
# Sweeps: N_CHUNKS x N_MEM_TOKENS x LR
# SIGMA is fixed at 1.0 (see ablation 3 in CLAUDE.md for sigma sweep).
# ENCODER_TYPE is "conv" (see ablation 2 for conv vs linear comparison).
#
# Usage:
#   bash scripts/h-tok/run_adaptive_rmt_on_kv_retrieval.sh
#   NP=4 bash scripts/h-tok/run_adaptive_rmt_on_kv_retrieval.sh   # multi-GPU
set -e

NP=${NP:-1}
TBS=64
PER_DEVICE_BATCH_SIZE=64
GRAD_ACC_STEPS=$(( TBS / (PER_DEVICE_BATCH_SIZE * NP) ))

# Model architecture (identical to fixed baseline)
L=4
H=4
D=128
BASE_MODEL=llama

# KV task
K=2
V=2
N_PAIRS=16
TOKENIZER_PATH="./tokenizers/kv_alphabet_62/"

# Chunker defaults
SIGMA=1.0
ENCODER_TYPE=conv

# Fixed training budget
ITERS=200000

DATA_PATH_NAME="N${N_PAIRS}-K${K}V${V}-V62_1M"

for N_CHUNKS in 1 2 4 8 16; do
  for N in 1 2; do
    for N_MEM_TOKENS in 4 8 32 64; do
      for LR in 1e-03 3e-04 5e-05; do

        RUN_NAME=adaptive_rmt_${BASE_MODEL}_L${L}H${H}D${D}_mem${N_MEM_TOKENS}_N${N_PAIRS}_K${N_CHUNKS}_sigma${SIGMA}_${ENCODER_TYPE}_bs${TBS}_lr${LR}
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
          --use_adaptive_chunking true \
          --n_chunks $N_CHUNKS \
          --chunker_sigma $SIGMA \
          --chunker_encoder_type $ENCODER_TYPE \
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

echo "Done: adaptive RMT sweep"
