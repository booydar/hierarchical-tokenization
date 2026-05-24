# Test Time Gradient Descend for Memory Update

This repository contains small experiments around "test time" gradient updates for key--value retrieval tasks. The main goal is to train compact GPT style models (both vanilla and models with an adaptive memory) to recover values that appear in the context.


## Intro
Large-context transformers pay a **quadratic cost** every time they reread long prompts.

Our goal is to compress those prompts into a **small, writable parameter block `[mem]`** that we update with a few gradient steps at test time, then drop the original text entirely.

### How it works

| Phase | What happens | N_iters | Input size |
|-------|--------------|--------------|------------|
| **Write (inner loop, *K* steps)** | Show the context, compute an LM loss **L<sub>inner</sub>**, update **`[mem]` only** | *K* | `[mem]` + `context` |
| **Read (outer loop)** | Discard the context; answer the query with the **updated `[mem]`** and compute **L<sub>outer</sub>** | 1 |  `query` |

*Back-propagating L<sub>outer</sub> meta-trains both the Transformer weights θ and the **initial memory `[mem]_0`**, so the model learns how to “write quickly.”*

### Gradient-flow modes

| Flag | What gradients reach `[mem]_0`? | Extra VRAM cost | Typical use-case |
|------|---------------------------------|-----------------|------------------|
| `none` (“Frozen”) | **None** (detach) | None | Baseline sanity check |
| `first` (“1st-order”) | Straight-through, no Hessian term | None | Fast runs, XX% of full accuracy |
| `second` (“2nd-order”) | Full MAML (keeps full graph through the *K* inner steps) | **≈ K × activation-memory** (parameters are shared; what multiplies is the *activations* for each inner forward/backward) | Highest accuracy when GPU RAM is sufficient |


## Prerequisites

* Python 3.11
* [conda](https://docs.conda.io/en/latest/) for environment management

Create an environment using the provided YAML file:

```bash
conda env create -f conda_env.yaml
conda activate /home/jovyan/kuratov/envs/py311_pt2.6_cu12.4  # or the path printed by conda
```

Accelerate is configured via `accelerate.yaml`. The default configuration uses BF16 precision and a single process.

## Dataset generation

Datasets consist of sequences containing random text segments with embedded `!key:value!` pairs. The last segment queries one of the previous keys (e.g. `?!K:`) and the model must output the corresponding value.

To generate a dataset run the notebook `notebooks/dump_dataset.ipynb`. It relies on `kv_dataset_utils.generate_sequence` to create individual samples and dumps them using Hugging Face `datasets`. The resulting directory will be saved under `./data/<DATASET_NAME>` where `DATASET_NAME` encodes generation parameters, for example `N10-K4V4-S4(32-64)_1M`.

## Training

Two entry points are provided:

* `run_gpt2_on_kv_retrieval.py` &ndash; trains a standard causal LM.
* `run_gradmemgpt_on_kv_retrieval.py` &ndash; trains a small LM with writable memory (see `grad_memgpt.py`).

Both scripts accept the same arguments (batch size, number of layers, dataset path, etc.). They should be launched through `accelerate`:

```bash
accelerate launch --config_file accelerate.yaml \
  run_gpt2_on_kv_retrieval.py \
  --exp_path ./runs/gpt2_example \
  --per_device_batch_size 64 \
  --data_path ./data/N10-K4V4-S4(32-64)_1M
```

```bash
accelerate launch --config_file accelerate.yaml \
  run_gradmemgpt_on_kv_retrieval.py \
  --exp_path ./runs/gradmem_example \
  --per_device_batch_size 64 \
  --data_path ./data/N10-K4V4-S4(32-64)_1M
```

The scripts log metrics and save checkpoints to the directory specified via `--exp_path`.

## Hierarchical tokenization (learned segmentation + RMT)

This repo implements **content-adaptive chunking** for Recurrent Memory Transformer (RMT) on the KV-retrieval task. Instead of splitting the context into fixed-size windows, a chunker learns where segment boundaries fall; RMT then runs one recurrent step per segment.

There are **two entry points** (different chunker designs):

| Script | Chunker | Model class | Typical use |
|--------|---------|-------------|-------------|
| `run_dynamic_rmt_on_kv_retrieval.py` | H-Net-style `RoutingModule` (cosine similarity + argmax boundaries) | `RMTForReasoningDynamicChunking` | STE + ratio loss; flat sequences via `collate_fn_dynamic` |
| `run_h_tok_on_kv_retrieval.py` | `DynamicChunker` (soft Gaussian assignment, conv/linear encoder) | `RMTForAdaptiveReasoning` | Stage-1 HD-RMT per `CLAUDE.md`; fixed `K` chunks |

Both reuse the same KV datasets under `./data/` and the tokenizer at `./tokenizers/kv_alphabet_62/`. See `CLAUDE.md` for design goals and ablations.

### Quick start: H-Net dynamic chunking (`run_dynamic_rmt_on_kv_retrieval.py`)

Single-GPU example (4 KV pairs, target ~4 tokens per segment):

```bash
conda activate <your-env>   # see Prerequisites
cd /path/to/hierarchical-tokenization

accelerate launch --config_file accelerate.yaml \
  run_dynamic_rmt_on_kv_retrieval.py \
  --exp_path ./runs/dyn_chunk_N4_kv \
  --per_device_batch_size 64 \
  --data_path ./data/P4-K4V4-S4(32-64)_1M \
  --tokenizer_path ./tokenizers/kv_alphabet_62/ \
  --base_model gpt2 \
  --n_layer 4 --n_head 4 --n_embd 128 \
  --n_pairs 4 --n_keys 2 --n_values 2 \
  --n_mem_tokens 4 \
  --max_n_segments 96 \
  --chunker_compression_ratio 4.0 \
  --chunker_aux_loss_weight 0.01 \
  --chunker_lr_multiplier 2.0 \
  --learning_rate 1e-4 \
  --max_steps 50000 \
  --eval_steps 200 \
  --logging_steps 50
```

Multi-GPU (e.g. 6 GPUs, effective batch = `per_device_batch_size × num_gpus`):

```bash
accelerate launch \
  --config_file accelerate.yaml \
  --num_processes 6 \
  run_dynamic_rmt_on_kv_retrieval.py \
  --exp_path ./runs/dyn_chunk_N8_kv \
  --per_device_batch_size 16 \
  --data_path ./data/N8-K4V4-S4(32-64)_1M \
  --n_pairs 8 --n_mem_tokens 8 \
  --chunker_compression_ratio 8.0 \
  ...
```

If `./data/...` is missing, the script **generates** a dataset on first run using `kv_dataset_utils.generate_sequence` (controlled by `--n_pairs`, `--n_keys`, `--n_values`).

**Query / QT segment.** The model finds the first ``?`` token (`RMTConfig.query_token_id`) and starts the final segment there. Default `split_query_target_segments=False` keeps **query + target in one segment** (same as fixed-segment RMT); only context boundaries are learned. Set `split_query_target_segments=True` for a separate target segment. For inference, call `model.generate` on **`context + query` only**.

**Chunker-specific flags**

| Flag | Meaning |
|------|---------|
| `chunker_compression_ratio` | Target average segment length `N` (tokens). Ratio loss pulls boundary rate toward `1/N`. Example: `4.0` ≈ one boundary every 4 tokens. |
| `chunker_aux_loss_weight` | λ in `loss = lm_loss + λ * ratio_loss` (default `0.01`). Set `0` to disable ratio loss (not recommended). |
| `chunker_ratio_loss_exclude_query_start` | If `true` (default), ratio loss ignores the forced query-start boundary at `?` (like position 0). Set `false` to include it in the global `1/N` target. |
| `split_query_target_segments` | If `false` (default), one final RMT segment for query+target (original RMT parity). If `true`, target is a separate segment. |
| `chunker_lr_multiplier` | AdamW LR for `routing_module` = `learning_rate × multiplier` (default `2.0`). |

**Training diagnostics**

- TensorBoard: `tensorboard --logdir runs/<exp_name>`
- Logged scalars: `lm_loss`, `ratio_loss`, `mean_p_boundary`, `empirical_boundary_rate`
- On the first real training step, a log line confirms routing-module gradients (`Sanity check: routing module gradient OK`)

**Resume from checkpoint**

```bash
  --model_cpt ./runs/dyn_chunk_N4_kv
```

The script picks the latest `checkpoint-*` and loads `model.safetensors`.

### Quick start: soft DynamicChunker (`run_h_tok_on_kv_retrieval.py`)

Fixed vs adaptive ablation scripts live under `scripts/h-tok/`:

```bash
# Adaptive (learned boundaries, fixed K chunks)
bash scripts/h-tok/run_adaptive_rmt_on_kv_retrieval.sh

# Fixed-stride baseline (same data, pairs_per_segment sweep)
bash scripts/h-tok/run_fixed_rmt_on_kv_retrieval.sh

# Multi-GPU
NP=4 bash scripts/h-tok/run_adaptive_rmt_on_kv_retrieval.sh
```

Manual launch:

```bash
accelerate launch --config_file accelerate.yaml \
  run_h_tok_on_kv_retrieval.py \
  --exp_path ./runs-htok/my_adaptive_run \
  --per_device_batch_size 64 \
  --data_path ./data/N16-K2V2-V62_1M \
  --use_adaptive_chunking true \
  --n_chunks 8 \
  --chunker_sigma 1.0 \
  --chunker_encoder_type conv \
  --n_mem_tokens 8 \
  ...
```

Key flags: `n_chunks` (fixed number of pooled chunks), `chunker_sigma` (assignment sharpness), `chunker_encoder_type` (`conv` or `linear`), `hard_inference` (hard boundaries at eval).

### Inspect learned segment boundaries

After training dynamic-RMT checkpoints under `runs/dyn_chunk_*`:

```bash
PYTHONPATH=. python scripts/visualize_learned_segments.py \
  --runs_root runs \
  --n_examples 2 \
  --pair_counts 2 4 8
```

This loads each run’s latest checkpoint, runs the routing module on validation samples, and prints per-segment token spans and boundary probabilities (in-distribution vs other pair counts).

### Comparison to fixed-segment RMT

| | Fixed RMT | Dynamic RMT (`run_dynamic_rmt`) |
|--|-----------|----------------------------------|
| Entry script | `run_original_rmt_on_kv_retrieval-v3-gen.py` | `run_dynamic_rmt_on_kv_retrieval.py` |
| Dataloader | Pre-segmented dicts per window | Flat `input_ids` + `labels=-100` on context |
| Boundaries | `pairs_per_segment` / chunk size | Learned `RoutingModule` + ratio loss |

For fair comparison, set `chunker_compression_ratio` to roughly the same token budget you would use for a fixed chunk size on the same task (e.g. ratio `8` when the fixed baseline uses ~8 tokens per segment).

