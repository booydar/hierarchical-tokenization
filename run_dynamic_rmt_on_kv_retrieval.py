"""Train RMT with H-Net-style learned segmentation on KV retrieval.

This is the dynamic-chunking counterpart of
``run_original_rmt_on_kv_retrieval-v3-gen.py``. Three differences:

1. The model is :class:`RMTForReasoningDynamicChunking`. The wrapper inside
   it owns a learnable :class:`RoutingModule` and decides per-token where
   segment boundaries fall.

2. ``collate_fn_dynamic`` produces a *flat* batch
   (``input_ids``, ``attention_mask``, ``labels``, ``labels_mask``) rather
   than pre-segmented dicts. Context tokens get ``label = -100`` so the per-
   segment cross entropy in ``process_outputs`` ignores them — including for
   "mixed" segments that straddle the context/query boundary.

3. ``DynamicChunkingTrainer`` adds:
       * a 2× learning-rate group for the routing module (chunker is trained
         from scratch while the base LM may be initialised from a
         checkpoint — see ``CLAUDE.md``);
       * surfacing of ``lm_loss`` / ``ratio_loss`` / mean ``boundary_prob``
         to TensorBoard via ``self.log`` from ``compute_loss``;
       * a one-shot sanity check on the very first training batch that
         verifies the chunker is actually receiving gradient.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

import datasets
import accelerate
import transformers
from transformers import (
    AutoConfig, AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback, TrainerCallback,
    HfArgumentParser,
)


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_lvl = logging.INFO
logging.basicConfig(format=logger_fmt, level=log_lvl)
logger = logging.getLogger('')

logger.info(f"CUDA DEVICE COUNT: {torch.cuda.device_count()}")


# ---------------------------------------------------------------------------
# Dataloader
# ---------------------------------------------------------------------------


def collate_fn_dynamic(batch):
    """Flat-sequence collator for :class:`RecurrentWrapperDynamicChunking`.

    Output shapes (after padding to the longest example in the batch):

        input_ids       [B, L]   long
        attention_mask  [B, L]   long
        labels          [B, L]   long, -100 on context+query, real IDs on target
        labels_mask     [B, L]   bool, True only on target positions

    The chunker decides segment boundaries at training time. With the -100
    convention plus ``labels_mask``, per-segment CE in ``process_outputs``
    correctly ignores all non-target positions regardless of how the chunker
    happens to split a sequence.
    """
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    examples = []
    for sample in batch:
        # Memory-task augmentation kept identical to the fixed-segment
        # baseline so the two runs are directly comparable.
        perform_memory_task = torch.rand(1) < args.memory_task_freq
        if perform_memory_task and args.memory_task == "reconstruct":
            query = '!?'
            target = '!?' + sample['context'][2:-2]
        elif perform_memory_task and args.memory_task == "continue":
            ctx = sample['context']
            qs = torch.randint(
                0, len(ctx) - args.memory_key_size - args.memory_value_size - 4, (1,)
            )
            query = '!?' + ctx[qs:qs + args.memory_key_size]
            target = '!?' + ctx[
                qs + args.memory_key_size:
                qs + args.memory_key_size + args.memory_value_size
            ]
        else:
            query = sample['query']
            target = sample['target']

        ctx_ids = encode(sample['context'])
        q_ids = encode(query)
        t_ids = encode(target)
        full_ids = ctx_ids + q_ids + t_ids

        # -100 on everything except target tokens. The per-segment CE
        # filter (`shift_labels != -100`) handles this automatically.
        labels = [-100] * (len(ctx_ids) + len(q_ids)) + t_ids
        # labels_mask is a *belt and braces* extra mask that some
        # downstream tooling (e.g. compute_metrics) consults. It carries
        # the same information as `labels != -100` here.
        labels_mask = [False] * (len(ctx_ids) + len(q_ids)) + [True] * len(t_ids)

        examples.append({
            'input_ids':      torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.ones(len(full_ids), dtype=torch.long),
            'labels':         torch.tensor(labels, dtype=torch.long),
            'labels_mask':    torch.tensor(labels_mask, dtype=torch.bool),
        })

    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        pad_id = 0

    return {
        'input_ids':      pad_sequence([e['input_ids']      for e in examples], batch_first=True, padding_value=pad_id),
        'attention_mask': pad_sequence([e['attention_mask'] for e in examples], batch_first=True, padding_value=0),
        'labels':         pad_sequence([e['labels']         for e in examples], batch_first=True, padding_value=-100),
        'labels_mask':    pad_sequence([e['labels_mask']    for e in examples], batch_first=True, padding_value=False),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics_fn(eval_pred, ignore_token_ids, tokenizer):
    """Token-accuracy + exact-match. Same as the fixed-segment script,
    minus the memory-task branch (kept for parity)."""
    predictions, labels, inputs = eval_pred.predictions, eval_pred.label_ids, eval_pred.inputs
    # Belt-and-braces: with the dynamic-chunking model we set
    # `keys_to_ignore_at_inference` on the config so HF Trainer unpacks
    # `predictions` to a bare tensor. If for any reason it didn't
    # (e.g. older HF Trainer, or extra keys we add later), `predictions`
    # will be a tuple; pull out the logits ourselves.
    logits = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    # Same belt-and-braces for labels: if a future forward() ever picks up a
    # second `*label*`-named kwarg, `label_ids` would become a tuple again.
    labels = labels[0] if isinstance(labels, (tuple, list)) else labels
    # Teacher-forcing shift.
    logits = logits[..., :-1, :]
    labels = labels[..., 1:]
    preds = np.argmax(logits, axis=-1)

    mask = (labels != -100)
    for t_id in ignore_token_ids:
        mask &= (labels != t_id)

    accuracy = (preds[mask] == labels[mask]).mean() if mask.any() else 0.0

    decoded_labels = [
        tokenizer.decode(label[label != -100], skip_special_tokens=True).replace(' ', '')
        for label in labels
    ]
    memory_task_mask = [decoded_labels[i][:2] == '!?' for i in range(len(decoded_labels))]

    def _emp(predicate):
        vals = [
            np.all(preds[i][mask[i]] == labels[i][mask[i]])
            for i in range(len(preds))
            if np.any(mask[i]) and predicate(i)
        ]
        return float(np.mean(vals)) if vals else 0.0

    exact_match_memory_task = _emp(lambda i: memory_task_mask[i])
    exact_match_base = _emp(lambda i: not memory_task_mask[i])

    n_samples = 5
    for pred, label, inp in zip(preds[:n_samples], labels[:n_samples], inputs[:n_samples]):
        m = (label != -100)
        pred = pred[m]
        inp = inp.copy(); inp[inp == -100] = 0
        label = label.copy(); label[label == -100] = 0
        print('i:', tokenizer.decode(inp,   skip_special_tokens=True).replace(' ', ''))
        print('p:', tokenizer.decode(pred,  skip_special_tokens=True).replace(' ', ''))
        print('t:', tokenizer.decode(label, skip_special_tokens=True).replace(' ', ''))
        print('-' * 50)

    res = {
        "token_accuracy": float(accuracy),
        "exact_match_base": exact_match_base,
    }
    if args.memory_task is not None:
        res[f"exact_match_{args.memory_task}"] = exact_match_memory_task
    return res


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class StopOnMetricValue(TrainerCallback):
    """Stop training as soon as ``metric_name`` reaches a threshold."""
    def __init__(self, metric_name: str, value: float, higher_is_better: bool = True):
        self.metric_name = metric_name
        self.value = value
        self.higher_is_better = higher_is_better

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = self.metric_name if self.metric_name.startswith("eval_") \
            else f"eval_{self.metric_name}"
        metric_value = metrics.get(metric_to_check)
        if metric_value is None:
            return
        op = np.greater_equal if self.higher_is_better else np.less_equal
        if op(metric_value, self.value):
            control.should_training_stop = True
            logger.info(
                f'metric {self.metric_name}={metric_value:.4f} hit threshold {self.value:.4f}; stopping.'
            )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DynamicChunkingTrainer(Trainer):
    """Extends :class:`Trainer` with the three dynamic-chunking conveniences.

    1. ``create_optimizer`` puts the routing module in its own param group at
       ``chunker_lr_multiplier × base_lr``. This is what ``CLAUDE.md``
       recommends: the chunker is trained from scratch and benefits from a
       higher LR than the base LM.
    2. ``compute_loss`` surfaces ``lm_loss`` / ``ratio_loss`` / mean
       ``boundary_prob`` / empirical boundary rate to ``self.log`` so they
       appear in TensorBoard alongside the regular ``loss``.
    3. A one-shot sanity check on the first training batch verifies that the
       routing module is actually receiving gradient (it's silent failure
       otherwise — the chunker stays at its identity init and you only
       notice after thousands of wasted steps).
    """

    def __init__(self, *args, chunker_lr_multiplier: float = 2.0,
                 sanity_check_first_batch: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunker_lr_multiplier = chunker_lr_multiplier
        self._sanity_check_done = not sanity_check_first_batch

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        # Same trick the fixed-segment baseline uses to keep final LR > 0.
        num_training_steps = int(num_training_steps / 0.9)
        return super().create_scheduler(num_training_steps, optimizer)

    def create_optimizer(self):
        """Two parameter groups: routing module at higher LR, everything else
        at the base LR (with the usual no-decay-on-bias-and-LN exclusion)."""
        if self.optimizer is not None:
            return self.optimizer

        # The routing module lives under model.rmt.routing_module.
        chunker_params = list(self.model.rmt.routing_module.parameters())
        chunker_param_ids = {id(p) for p in chunker_params}

        decay_params, nodecay_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad or id(p) in chunker_param_ids:
                continue
            if p.ndim < 2 or n.endswith('.bias') or 'layernorm' in n.lower() or 'layer_norm' in n.lower():
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        lr = self.args.learning_rate
        wd = self.args.weight_decay
        groups = [
            {'params': decay_params,   'lr': lr,                                'weight_decay': wd},
            {'params': nodecay_params, 'lr': lr,                                'weight_decay': 0.0},
            {'params': chunker_params, 'lr': lr * self.chunker_lr_multiplier,   'weight_decay': 0.0,
             'name':   'routing_module'},
        ]
        # Drop empty groups; otherwise some optimizers (e.g. AdamW with
        # fused=True) complain.
        groups = [g for g in groups if len(g['params']) > 0]

        self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Run the model, log the extra diagnostics, and hand HF back the
        standard ``loss`` it wants for ``.backward()``."""
        outputs = model(**inputs)
        loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

        # Log auxiliary signals. We only call ``self.log`` here when the
        # trainer is in a "should log" step to keep TB clean.
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            logs: Dict[str, float] = {}
            # nn.DataParallel turns per-replica scalars into a [n_gpu] vector
            # when gathering. Reduce defensively before `.item()` so we don't
            # crash with "Tensor with N elements cannot be converted to Scalar".
            def _scalar(t: torch.Tensor) -> float:
                return (t.mean() if t.dim() > 0 else t).detach().float().item()

            lm_loss = outputs.get('lm_loss') if isinstance(outputs, dict) else None
            if isinstance(lm_loss, torch.Tensor):
                logs['lm_loss'] = _scalar(lm_loss)
            ratio_loss = outputs.get('ratio_loss') if isinstance(outputs, dict) else None
            if isinstance(ratio_loss, torch.Tensor):
                logs['ratio_loss'] = _scalar(ratio_loss)
            boundary_prob = outputs.get('boundary_prob') if isinstance(outputs, dict) else None
            if isinstance(boundary_prob, torch.Tensor):
                # Mean P(boundary) over real tokens (skip the forced first).
                attn = inputs.get('attention_mask')
                if attn is not None:
                    valid = attn.bool().float().clone()
                    valid[:, 0] = 0.0
                    denom = valid.sum().clamp(min=1.0)
                    logs['mean_p_boundary'] = (
                        (boundary_prob[..., 1] * valid).sum() / denom
                    ).detach().float().item()
            boundary_mask = outputs.get('boundary_mask') if isinstance(outputs, dict) else None
            if isinstance(boundary_mask, torch.Tensor):
                attn = inputs.get('attention_mask')
                if attn is not None:
                    valid = attn.bool().float().clone()
                    valid[:, 0] = 0.0
                    denom = valid.sum().clamp(min=1.0)
                    logs['empirical_boundary_rate'] = (
                        (boundary_mask & attn.bool()).float() * valid
                    ).sum().div(denom).detach().float().item()
            if logs:
                self.log(logs)

        # Only attempt the gradient sanity check during training (when the
        # loss has a grad graph). `eval_on_start=True` and any
        # eval-during-training pass call `compute_loss` under `no_grad`,
        # at which point `loss.grad_fn is None` and `backward()` would
        # raise. We also keep `_sanity_check_done` False until we've
        # actually managed to run the check, so the first real training
        # step is the one that gets verified.
        if not self._sanity_check_done and loss.requires_grad and loss.grad_fn is not None:
            self._sanity_check_done = self._run_first_batch_sanity_check(loss, outputs)

        return (loss, outputs) if return_outputs else loss

    def _run_first_batch_sanity_check(self, loss, outputs):
        """Verify gradient is reaching the routing module.

        Returns
        -------
        bool
            ``True`` iff the check actually ran (regardless of pass/fail).
            ``False`` means an exception aborted it and we should retry on
            the next training-mode batch.

        We run a *non-destructive* backward (``retain_graph=True``) so the
        real training step that follows still works. Caveat: this doubles
        backward cost for the very first step only.
        """
        try:
            # Under nn.DataParallel, `loss` arriving here is a vector of shape
            # [n_gpu] (one scalar per replica). HF Trainer reduces it with
            # `.mean()` *after* compute_loss returns; we have to do the same
            # locally before calling backward, otherwise PyTorch raises
            # "grad can be implicitly created only for scalar outputs".
            sanity_loss = loss.mean() if loss.dim() > 0 else loss
            sanity_loss.backward(retain_graph=True)
            g = self.model.rmt.routing_module.q_proj_layer.weight.grad
            if g is None or float(g.abs().mean()) == 0.0:
                logger.warning(
                    "Sanity check: routing module did NOT receive gradient on "
                    "the first batch. Check that model.training is True and "
                    "that the STE block in RecurrentWrapperDynamicChunking.forward "
                    "ran. Training will continue but the chunker is frozen."
                )
            else:
                logger.info(
                    f"Sanity check: routing module gradient OK "
                    f"(mean |grad|={float(g.abs().mean()):.3e})."
                )
            # Reset grads so HF Trainer's own backward + step is clean.
            self.model.zero_grad(set_to_none=True)
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Sanity check failed to run: {e}")
            return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class ExperimentArgs:
    exp_path: str = field()
    per_device_batch_size: int = field()
    data_path: str = field(default='./data/N2-K4V4-S4(32-64)_1M')
    tokenizer_path: str = field(default='./tokenizers/kv_alphabet_62/')

    gradient_accumulation_steps: Optional[int] = field(default=1)
    total_batch_size: Optional[int] = field(default=None)
    metric_for_best_model: Optional[str] = field(default='token_accuracy')
    warmup_steps: Optional[int] = field(default=1000)
    max_steps: Optional[int] = field(default=50000)
    logging_steps: Optional[int] = field(default=100)
    eval_steps: Optional[int] = field(default=100)
    weight_decay: Optional[float] = field(default=0.0)
    learning_rate: Optional[float] = field(default=1e-04)
    lr_scheduler_type: Optional[str] = field(default='constant_with_warmup')
    early_stopping_patience: Optional[int] = field(default=50)
    seed: Optional[int] = field(default=142)

    base_model: Optional[str] = field(default='gpt2')
    n_layer: Optional[int] = field(default=4)
    n_head: Optional[int] = field(default=4)
    n_embd: Optional[int] = field(default=128)

    # RMT parameters.
    n_mem_tokens: Optional[int] = field(default=8)
    max_n_segments: Optional[int] = field(default=32)
    k2: Optional[int] = field(default=-1)
    memory_task_freq: Optional[float] = field(default=0.0)
    memory_task: Optional[str] = field(default=None)
    memory_key_size: Optional[int] = field(default=4)
    memory_value_size: Optional[int] = field(default=4)
    model_cpt: Optional[str] = field(default=None)

    # Dataset generation parameters.
    n_pairs: Optional[int] = field(default=None)
    n_keys: Optional[int] = field(default=None)
    n_values: Optional[int] = field(default=None)

    # ---- new: dynamic-chunking knobs ----
    chunker_compression_ratio: Optional[float] = field(default=8.0)
    chunker_aux_loss_weight: Optional[float] = field(default=0.01)
    chunker_lr_multiplier: Optional[float] = field(default=2.0)


if __name__ == '__main__':
    parser = HfArgumentParser(ExperimentArgs)
    args = parser.parse_args_into_dataclasses()[0]

    accel = accelerate.Accelerator()
    from accelerate.logging import get_logger
    logger = get_logger('')
    transformers.utils.logging.set_verbosity(log_lvl)

    logger.info(f'num processes: {accel.num_processes}')
    logger.info(f'mixed precision: {accel.mixed_precision}')
    logger.info(f'accelerator state: {accel.state}')

    if accel.is_main_process:
        Path(args.exp_path).mkdir(parents=True, exist_ok=True)
        json.dump({'cli_args': dict(vars(args))},
                  open(os.path.join(args.exp_path, 'config.json'), 'w'), indent=4)
        logger.info(f'saved experiment configuration to {args.exp_path}')

    # ---------------- tokenizer + base config ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.base_model == 'gpt2':
        config = AutoConfig.from_pretrained('gpt2')
        config.n_layer, config.n_head, config.n_embd = args.n_layer, args.n_head, args.n_embd
    elif args.base_model == 'pythia':
        config = AutoConfig.from_pretrained('EleutherAI/pythia-160m')
        config.num_hidden_layers = args.n_layer
        config.num_attention_heads = args.n_head
        config.hidden_size = args.n_embd
        config.intermediate_size = config.hidden_size * 4
    elif args.base_model == 'llama':
        config = AutoConfig.from_pretrained('NousResearch/Llama-3.2-1B')
        config.num_hidden_layers = args.n_layer
        config.num_attention_heads = args.n_head
        config.num_key_value_heads = args.n_head
        config.hidden_size = args.n_embd
        config.head_dim = config.hidden_size // config.num_attention_heads
        config.intermediate_size = config.hidden_size * 4
    else:
        raise ValueError(f'Unsupported base model: {args.base_model}')

    config.torch_dtype = "float32"
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')
    config.bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
    config.eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')

    # ---------------- RMT + dynamic-chunking model ----------------
    from modeling_rmt.huggingface import RMTConfig, RMTForReasoningDynamicChunking

    rmt_config = RMTConfig()
    rmt_config.base_model_config = config
    rmt_config.num_mem_tokens = args.n_mem_tokens
    rmt_config.max_n_segments = args.max_n_segments
    rmt_config.k2 = args.k2
    rmt_config.bos_token_id = tokenizer.convert_tokens_to_ids('[BOS]')
    rmt_config.eos_token_id = tokenizer.convert_tokens_to_ids('[EOS]')
    rmt_config.chunker_compression_ratio = args.chunker_compression_ratio
    rmt_config.chunker_aux_loss_weight = args.chunker_aux_loss_weight

    model = RMTForReasoningDynamicChunking(rmt_config)
    # Default main_input_name is 'input_ids', which is what our flat collate
    # produces. We don't need to override it here.

    if args.model_cpt and args.model_cpt != 'None':
        # Mirror the checkpoint-search logic of the fixed-segment script.
        cpt_path = args.model_cpt
        use_safetensors = False
        if os.path.isdir(cpt_path):
            dir_files = os.listdir(cpt_path)
            if "model_best" in dir_files:
                cand = os.path.join(cpt_path, "model_best", "pytorch_model.bin")
                if os.path.exists(cand):
                    cpt_path = cand
                else:
                    cand_st = os.path.join(cpt_path, "model_best", "model.safetensors")
                    if os.path.exists(cand_st):
                        cpt_path = cand_st; use_safetensors = True
                    else:
                        raise FileNotFoundError(f"No checkpoint in {cpt_path}/model_best")
            else:
                checkpoints = sorted([d for d in dir_files if d.startswith("checkpoint-")])
                if not checkpoints:
                    raise FileNotFoundError(f"No checkpoint- dir in {cpt_path}")
                ck_dir = os.path.join(cpt_path, checkpoints[-1])
                cand = os.path.join(ck_dir, "pytorch_model.bin")
                if os.path.exists(cand):
                    cpt_path = cand
                else:
                    cpt_path = os.path.join(ck_dir, "model.safetensors")
                    use_safetensors = True
        elif os.path.isfile(cpt_path):
            use_safetensors = cpt_path.endswith(".safetensors")
        else:
            raise FileNotFoundError(f"Checkpoint path does not exist: {cpt_path}")

        if use_safetensors:
            from safetensors.torch import load_model
            load_model(model, cpt_path, device="cuda:0")
        else:
            cpt = torch.load(cpt_path, map_location='cpu')
            model.load_state_dict(cpt, strict=False)
        logger.info(f"Loaded model checkpoint from {cpt_path}")

    logger.info(f'model config: {model.config}')
    logger.info(f'model: {model}')

    # ---------------- dataset ----------------
    data_path = args.data_path
    try:
        dataset = datasets.load_from_disk(data_path)
        logger.info(f'Loaded dataset from {data_path}')
    except Exception as e:
        logger.info(f'Could not load dataset from {data_path}: {e}; generating fresh one.')
        from kv_dataset_utils import generate_sequence
        raw_samples = [
            generate_sequence(num_kv_pairs=args.n_pairs,
                              n_segments=1, min_segment_len=0, max_segment_len=0,
                              k_length=args.n_keys, v_length=args.n_values)
            for _ in range(1_005_000)
        ]
        import datasets as ds
        dataset = ds.Dataset.from_dict({
            'context': [s['context'] for s in raw_samples],
            'query':   [s['query']   for s in raw_samples],
            'target':  [s['target']  for s in raw_samples],
        })
        dataset = dataset.train_test_split(test_size=5_000, seed=args.seed)
        if "test" in dataset:
            dataset = datasets.DatasetDict({"train": dataset["train"], "valid": dataset["test"]})
        dataset.save_to_disk(data_path)
        logger.info(f'Generated dataset with {len(dataset["train"])} train / {len(dataset["valid"])} valid samples')

    ignore_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ['!', '|']]

    def compute_metrics(eval_preds):
        return compute_metrics_fn(eval_preds, ignore_token_ids, tokenizer)

    # ---------------- training arguments ----------------
    output_dir = Path(args.exp_path)
    if args.total_batch_size is None:
        args.total_batch_size = args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps
    else:
        assert args.total_batch_size == \
            args.per_device_batch_size * accel.num_processes * args.gradient_accumulation_steps

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

        eval_strategy='steps',
        save_strategy='steps',
        save_steps=args.eval_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        report_to='tensorboard',
        metric_for_best_model=args.metric_for_best_model,
        load_best_model_at_end=True,
        eval_on_start=True,
        greater_is_better=True,
        remove_unused_columns=False,
        include_num_input_tokens_seen=False,
        include_for_metrics=['inputs'],
        save_total_limit=1,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        seed=args.seed,
        # Pin the supervision target name: our model.forward also declares a
        # `labels_mask` argument, and HuggingFace Trainer's `find_labels`
        # auto-detection would otherwise pick up *both* "labels" and
        # "labels_mask" as labels — turning `eval_pred.label_ids` into a
        # 2-tuple and breaking any compute_metrics that expects a tensor.
        label_names=['labels'],
    )

    trainer = DynamicChunkingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        data_collator=collate_fn_dynamic,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            StopOnMetricValue(metric_name='exact_match_base', value=0.99, higher_is_better=True),
        ],
        chunker_lr_multiplier=args.chunker_lr_multiplier,
        sanity_check_first_batch=True,
    )

    trainer.train()
    logger.info('training done. running final evaluation...')
    metrics = trainer.evaluate(dataset['valid'])
    logger.info(f'{metrics}')
    trainer.save_metrics(split='all', metrics=metrics)
