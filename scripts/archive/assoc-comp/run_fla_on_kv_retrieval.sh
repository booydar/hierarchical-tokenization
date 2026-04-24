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

K=2
V=2
TOKENIZER_PATH="./tokenizers/kv_alphabet_62/"

# "gated_delta_net": "GatedDeltaNet",
#     "delta_net": "DeltaNet",
#     "gla": "GatedLinearAttention",
#     "linear_attention": "LinearAttention",
#     "hgrn": "HGRNAttention",
#     "hgrn2": "HGRN2Attention",
#     "rwkv6": "RWKV6Attention",
#     "rwkv7": "RWKV7Attention",
#     "mamba": "Mamba",
#     "mamba2": "Mamba2",
#     "retention": "MultiScaleRetention",


for BASE_MODEL in delta_net linear_attention rwkv6 gated_delta_net; do
    for N_PAIRS in 4 8 16 32; do
        for LR in 1e-03 3e-04 1e-04 3e-03;
        do

        RUN_NAME="${BASE_MODEL}_L${L}H${H}D${D}"

        RUN_NAME=${RUN_NAME}_bs_${TBS}_lr_${LR}

        if [ -n "$ADAM_BETA2" ]; then
            RUN_NAME=${RUN_NAME}_b2_${ADAM_BETA2}
        fi
        DATA_NAME="N${N_PAIRS}-K${K}V${V}-V62_1M"
        DATA_PATH="./data/${DATA_NAME}"

        N_VALUES=(1)
        for N in "${N_VALUES[@]}"; do
            EXP_PATH="./runs/${DATA_NAME}/${RUN_NAME}/run_${N}"

            accelerate launch \
            --main_process_port $((29500+$TBS+$N+1)) \
            --num_processes $NP \
            --mixed_precision bf16 \
            --config_file accelerate.yaml \
            run_fla_on_kv_retrieval.py \
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
            --max_steps 100000 \
            --eval_steps 500 \
            --logging_steps 500 \
            --warmup_steps 10000 \
            --early_stopping_patience 50 \
            --seed $((142+$N))
        done
    done
  done
done

echo "Done"
