"""
HD-RMT Stage 1: KV Retrieval training script.

Supports both ablation conditions via --use_adaptive_chunking:
  False → fixed-stride RMT baseline (RMTForReasoning)
  True  → adaptive chunking RMT    (RMTForAdaptiveReasoning)

Usage:
  See scripts/h-tok/run_fixed_rmt_on_kv_retrieval.sh
      scripts/h-tok/run_adaptive_rmt_on_kv_retrieval.sh
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import datasets
import numpy as np
import torch
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence

import accelerate
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger("")

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def split_context_into_segments(context, pairs_per_segment=None):
    if pairs_per_segment is None:
        return [context]
    clean_context = context[1:-2].strip()
    pairs = [f"!{p}!" for p in clean_context.split("!!")]
    segments = [pairs[i : i + pairs_per_segment] for i in range(0, len(pairs), pairs_per_segment)]
    return ["".join(s) for s in segments]


def collate_fn(batch):
    """
    Splits each sample into segments:
      Fixed mode:    context → N context segments (by pairs_per_segment) + 1 query segment
      Adaptive mode: context → 1 full context segment + 1 query segment
                     (model applies DynamicChunker internally)
    """
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    segments_batch = []
    for sample in batch:
        context = sample["context"]
        query = sample["query"]
        target = sample["target"]

        query_ids = encode(query)
        target_ids = encode(target)
        qt_ids = query_ids + target_ids

        if args.use_adaptive_chunking:
            # Full context as a single segment; no pre-splitting.
            context_ids = encode(context)
            context_segs = [
                {
                    "input_ids": torch.tensor(context_ids, dtype=torch.long),
                    "attention_mask": torch.ones(len(context_ids), dtype=torch.long),
                    "labels": torch.full((len(context_ids),), -100, dtype=torch.long),
                    "labels_mask": torch.zeros(len(context_ids), dtype=torch.bool),
                }
            ]
        else:
            context_parts = split_context_into_segments(
                context, pairs_per_segment=args.pairs_per_segment
            )
            context_segs = []
            for part in context_parts:
                part_ids = encode(part)
                context_segs.append(
                    {
                        "input_ids": torch.tensor(part_ids, dtype=torch.long),
                        "attention_mask": torch.ones(len(part_ids), dtype=torch.long),
                        "labels": torch.full((len(part_ids),), -100, dtype=torch.long),
                        "labels_mask": torch.zeros(len(part_ids), dtype=torch.bool),
                    }
                )

        # Query+target segment: loss on target tokens only.
        qt_input_ids = torch.tensor(qt_ids, dtype=torch.long)
        labels = torch.full((len(qt_ids),), -100, dtype=torch.long)
        labels_mask = torch.zeros(len(qt_ids), dtype=torch.bool)
        if len(target_ids) > 0:
            labels[-len(target_ids) :] = torch.tensor(target_ids, dtype=torch.long)
            labels_mask[-len(target_ids) - 1 :] = True

        query_seg = {
            "input_ids": qt_input_ids,
            "attention_mask": torch.ones(len(qt_ids), dtype=torch.long),
            "labels": labels,
            "labels_mask": labels_mask,
        }

        segments_batch.append(context_segs + [query_seg])

    # Pad across the batch, segment by segment.
    num_segments = len(segments_batch[0])
    id_pad = (
        tokenizer.pad_token_id
        if getattr(tokenizer, "pad_token_id", None) is not None
        else 0
    )

    batch_segments = []
    for i in range(num_segments):
        input_ids   = pad_sequence([s[i]["input_ids"]   for s in segments_batch], batch_first=True, padding_value=id_pad)
        attn_mask   = pad_sequence([s[i]["attention_mask"] for s in segments_batch], batch_first=True, padding_value=0)
        labels_t    = pad_sequence([s[i]["labels"]      for s in segments_batch], batch_first=True, padding_value=-100)
        labels_mask = pad_sequence([s[i]["labels_mask"] for s in segments_batch], batch_first=True, padding_value=False)
        batch_segments.append(
            {
                "input_ids": input_ids,
                "attention_mask": attn_mask,
                "labels": labels_t,
                "labels_mask": labels_mask,
            }
        )

    full_labels = torch.cat([s["labels"] for s in batch_segments], dim=1)
    return {"segments": batch_segments, "labels": full_labels}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    predictions, labels, inputs = (
        eval_pred.predictions,
        eval_pred.label_ids,
        eval_pred.inputs,
    )
    logits = predictions[..., :-1, :]
    labels = labels[..., 1:]
    preds = np.argmax(logits, axis=-1)

    mask = labels != -100
    for t_id in ignore_token_ids:
        mask &= labels != t_id

    accuracy = (preds[mask] == labels[mask]).mean()

    decoded_labels = [
        tokenizer.decode(label[label != -100], skip_special_tokens=True).replace(" ", "")
        for label in labels
    ]

    exact_match = np.mean(
        [
            np.all(preds[i][mask[i]] == labels[i][mask[i]])
            for i in range(len(labels))
            if np.any(mask[i])
        ]
    )

    n_samples = 5
    for pred, label, inp in zip(preds[:n_samples], labels[:n_samples], inputs[:n_samples]):
        m = label != -100
        pred_tok = pred[m]
        inp[inp == -100] = 0
        label[label == -100] = 0
        print("i:", tokenizer.decode(inp, skip_special_tokens=True).replace(" ", ""))
        print("p:", tokenizer.decode(pred_tok, skip_special_tokens=True).replace(" ", ""))
        print("t:", tokenizer.decode(label, skip_special_tokens=True).replace(" ", ""))
        print("-" * 50)

    return {
        "token_accuracy": float(accuracy),
        "exact_match": float(exact_match),
    }


# ---------------------------------------------------------------------------
# Trainer customizations
# ---------------------------------------------------------------------------

class StopOnMetricValue(TrainerCallback):
    def __init__(self, metric_name: str, value: float, higher_is_better: bool = True):
        self.metric_name = metric_name
        self.value = value
        self.higher_is_better = higher_is_better

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        key = self.metric_name if self.metric_name.startswith("eval_") else f"eval_{self.metric_name}"
        metric_value = metrics.get(key)
        if metric_value is None:
            return
        op = np.greater_equal if self.higher_is_better else np.less_equal
        if op(metric_value, self.value):
            control.should_training_stop = True
            logger.info(f"metric {self.metric_name}={metric_value:.4f} >= {self.value:.4f}, stopping.")


class CustomTrainer(Trainer):
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        # Avoid lr reaching zero at end of schedule.
        num_training_steps = int(num_training_steps / 0.9)
        return super().create_scheduler(num_training_steps, optimizer)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                logs["patience"] = cb.early_stopping_patience_counter
                break
        return super().log(logs, start_time=start_time)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    data_path: str = field(default="./data/N16-K2V2-V62_1M")
    tokenizer_path: str = field(default="./tokenizers/kv_alphabet_62/")
    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default="token_accuracy")
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=200000)
    logging_steps: Optional[int] = field(default=500)
    eval_steps: Optional[int] = field(default=500)
    weight_decay: Optional[float] = field(default=0.01)
    learning_rate: Optional[float] = field(default=1e-4)
    lr_scheduler_type: Optional[str] = field(default="constant_with_warmup")
    early_stopping_patience: Optional[int] = field(default=500)
    seed: Optional[int] = field(default=142)
    model_cpt: Optional[str] = field(default=None)

    # Base model architecture
    base_model: Optional[str] = field(default="llama")
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)

    # RMT parameters (shared between fixed and adaptive)
    n_mem_tokens: Optional[int] = field(default=8)
    max_n_segments: Optional[int] = field(default=20)

    # KV task parameters
    n_pairs: Optional[int] = field(default=16)
    n_keys: Optional[int] = field(default=2)
    n_values: Optional[int] = field(default=2)

    # Fixed chunking parameters (used when use_adaptive_chunking=False)
    pairs_per_segment: Optional[int] = field(default=None)

    # Adaptive chunking parameters (used when use_adaptive_chunking=True)
    use_adaptive_chunking: Optional[bool] = field(default=False)
    n_chunks: Optional[int] = field(default=4)
    chunker_sigma: Optional[float] = field(default=1.0)
    chunker_hidden_dim: Optional[int] = field(default=None)
    chunker_encoder_type: Optional[str] = field(default="conv")
    hard_inference: Optional[bool] = field(default=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger("")
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f"num processes: {accel.num_processes}")
    logger.info(f"mixed precision: {accel.mixed_precision}")
    logger.info(f"use_adaptive_chunking: {args.use_adaptive_chunking}")

    if accel.is_main_process:
        config_dict = {"cli_args": dict(vars(args))}
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump(config_dict, open(os.path.join(args.exp_path, "config.json"), "w"), indent=4)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Base model config
    if args.base_model == "gpt2":
        base_config = AutoConfig.from_pretrained("gpt2")
        base_config.n_layer = args.n_layer
        base_config.n_head = args.n_head
        base_config.n_embd = args.n_embd
    elif args.base_model == "pythia":
        base_config = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
        base_config.num_hidden_layers = args.n_layer
        base_config.num_attention_heads = args.n_head
        base_config.hidden_size = args.n_embd
        base_config.intermediate_size = args.n_embd * 4
    elif args.base_model == "llama":
        base_config = AutoConfig.from_pretrained("NousResearch/Llama-3.2-1B")
        base_config.num_hidden_layers = args.n_layer
        base_config.num_attention_heads = args.n_head
        base_config.num_key_value_heads = args.n_head
        base_config.hidden_size = args.n_embd
        base_config.head_dim = args.n_embd // args.n_head
        base_config.intermediate_size = args.n_embd * 4
    else:
        raise ValueError(f"Unsupported base model: {args.base_model!r}")

    base_config.torch_dtype = "float32"
    base_config.vocab_size = tokenizer.vocab_size
    base_config.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    base_config.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    base_config.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    # Model
    if args.use_adaptive_chunking:
        from modeling_rmt.huggingface_htok import AdaptiveRMTConfig, RMTForAdaptiveReasoning

        rmt_config = AdaptiveRMTConfig(
            base_model_config=base_config,
            num_mem_tokens=args.n_mem_tokens,
            max_n_segments=args.max_n_segments,
            think_token_id=tokenizer.convert_tokens_to_ids("[THINK]"),
            answer_token_id=tokenizer.convert_tokens_to_ids("[ANSWER]"),
            bos_token_id=base_config.bos_token_id,
            eos_token_id=base_config.eos_token_id,
            # Chunker params
            use_adaptive_chunking=True,
            n_chunks=args.n_chunks,
            chunker_hidden_dim=args.chunker_hidden_dim,
            chunker_sigma=args.chunker_sigma,
            chunker_encoder_type=args.chunker_encoder_type,
            hard_inference=args.hard_inference,
        )
        model = RMTForAdaptiveReasoning(rmt_config)
    else:
        from modeling_rmt.huggingface import RMTConfig, RMTForReasoning

        rmt_config = RMTConfig(
            base_model_config=base_config,
            num_mem_tokens=args.n_mem_tokens,
            max_n_segments=args.max_n_segments,
            think_token_id=tokenizer.convert_tokens_to_ids("[THINK]"),
            answer_token_id=tokenizer.convert_tokens_to_ids("[ANSWER]"),
            bos_token_id=base_config.bos_token_id,
            eos_token_id=base_config.eos_token_id,
        )
        model = RMTForReasoning(rmt_config)

    model.main_input_name = "labels"

    if args.model_cpt and args.model_cpt != "None":
        model_cpt_path = args.model_cpt
        use_safetensors = False
        if os.path.isdir(model_cpt_path):
            dir_files = os.listdir(model_cpt_path)
            if "model_best" in dir_files:
                candidate = os.path.join(model_cpt_path, "model_best", "pytorch_model.bin")
                if os.path.exists(candidate):
                    model_cpt_path = candidate
                else:
                    model_cpt_path = os.path.join(model_cpt_path, "model_best", "model.safetensors")
                    use_safetensors = True
            else:
                checkpoints = sorted(e for e in dir_files if e.startswith("checkpoint-"))
                checkpoint_dir = os.path.join(model_cpt_path, checkpoints[-1])
                candidate = os.path.join(checkpoint_dir, "pytorch_model.bin")
                if os.path.exists(candidate):
                    model_cpt_path = candidate
                else:
                    model_cpt_path = os.path.join(checkpoint_dir, "model.safetensors")
                    use_safetensors = True
        elif model_cpt_path.endswith(".safetensors"):
            use_safetensors = True

        if use_safetensors:
            from safetensors.torch import load_model
            load_model(model, model_cpt_path, device="cuda:0")
        else:
            cpt = torch.load(model_cpt_path, map_location="cpu")
            model.load_state_dict(cpt, strict=False)
        logger.info(f"Loaded checkpoint from {model_cpt_path}")

    logger.info(f"model:\n{model}")

    # Dataset
    data_path = args.data_path
    try:
        logger.info(f"Loading dataset from: {data_path}")
        dataset = datasets.load_from_disk(data_path)
        logger.info(f"Loaded {len(dataset['train'])} train / {len(dataset['valid'])} valid samples")
    except Exception as e:
        logger.info(f"Could not load dataset ({e}); generating...")
        from kv_dataset_utils import generate_sequence

        raw = [
            generate_sequence(
                num_kv_pairs=args.n_pairs,
                n_segments=1,
                min_segment_len=0,
                max_segment_len=0,
                k_length=args.n_keys,
                v_length=args.n_values,
            )
            for _ in range(1_005_000)
        ]
        import datasets as ds

        dataset = ds.Dataset.from_dict(
            {
                "context": [s["context"] for s in raw],
                "query": [s["query"] for s in raw],
                "target": [s["target"] for s in raw],
            }
        )
        dataset = dataset.train_test_split(test_size=5_000, seed=args.seed)
        dataset = datasets.DatasetDict({"train": dataset["train"], "valid": dataset["test"]})
        dataset.save_to_disk(data_path)
        logger.info(f"Generated and saved dataset to {data_path}")

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["!", "|"]]

    def compute_metrics(eval_preds):
        return compute_metrics_fn(eval_preds, ignore_token_ids, tokenizer)

    if args.total_batch_size is None:
        args.total_batch_size = (
            args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
        )

    output_dir = Path(args.exp_path)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(output_dir),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to="tensorboard",
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,
        include_for_metrics=["inputs"],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            StopOnMetricValue("exact_match", value=0.99, higher_is_better=True),
        ],
    )

    trainer.train()
    logger.info("Training done. Running final evaluation...")
    metrics = trainer.evaluate(dataset["valid"])
    logger.info(f"{metrics}")
    trainer.save_metrics(split="all", metrics=metrics)
