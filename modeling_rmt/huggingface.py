from __future__ import annotations

import math
import torch
from torch.nn import CrossEntropyLoss

from transformers import StoppingCriteria
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import warnings
from chunker.dynamic_chunker import RoutingModule, ChunkLayer


class RMTConfig(PretrainedConfig):
    model_type = "rmt"
    # HF Trainer's `prediction_step` reads this attribute and excludes the
    # listed keys before packing the model output into the `predictions`
    # tuple it hands to `compute_metrics`. Without this, dynamic chunking
    # adds extra tensors (`lm_loss`, `ratio_loss`, `boundary_prob`,
    # `boundary_mask`) and `predictions` becomes a tuple, breaking any
    # `compute_metrics` that expects a bare logits tensor.
    # Harmless for the fixed-segment baseline — these keys simply don't
    # appear in its output.
    keys_to_ignore_at_inference = [
        "lm_loss", "ratio_loss", "boundary_prob", "boundary_mask", "hidden_states",
    ]

    def __init__(self,
                 base_model_name="HuggingFaceTB/SmolLM2-135M",
                 base_model_config=None,
                 from_pretrained=None,
                 num_mem_tokens=16,
                 max_n_segments=10,
                 think_token_id=None,
                 answer_token_id=None,
                 bos_token_id=None,
                 eos_token_id=None,
                 # Dynamic-chunking knobs (only read by
                 # RecurrentWrapperDynamicChunking / RMTForReasoningDynamicChunking).
                 # Kept on the base config so they are saved with the
                 # checkpoint and visible to HuggingFace Trainer.
                 chunker_compression_ratio=None,
                 chunker_aux_loss_weight=0.01,
                 chunker_ratio_loss_exclude_query_start=True,
                 k2=-1,
                 query_token_id=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.base_model_config = base_model_config
        self.from_pretrained = from_pretrained
        self.num_mem_tokens = num_mem_tokens
        self.max_n_segments = max_n_segments
        self.think_token_id = think_token_id
        self.answer_token_id = answer_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.memory_cell_cls = "MemoryCell"
        self.recurrent_wrapper_cls = "RecurrentWrapperNoSegmentationGenerate"
        self.chunker_compression_ratio = chunker_compression_ratio
        self.chunker_aux_loss_weight = chunker_aux_loss_weight
        # When True, ratio-loss F/G averages exclude the forced query-start
        # boundary (like position 0) so compression is regularized in context only.
        self.chunker_ratio_loss_exclude_query_start = chunker_ratio_loss_exclude_query_start
        self.k2 = k2
        # Token id for ``?`` — used to locate the KV query (``?!key:``) and
        # force it into its own segment. Set from the task tokenizer.
        self.query_token_id = query_token_id

    def get(self, attr: str, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return default


class RMTForReasoning(PreTrainedModel):
    config_class = RMTConfig

    def __init__(self, config: RMTConfig, **kwargs):
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM
        if config.from_pretrained:
            base_model = AutoModelForCausalLM.from_pretrained(config.from_pretrained)
        else:
            if config.base_model_config is None:
                base_config = AutoConfig.from_pretrained(config.base_model_name)
            else:
                base_config = config.base_model_config
            base_model = AutoModelForCausalLM.from_config(base_config)

        self.rmt_config = config
        memory_cell = MemoryCell(base_model, num_mem_tokens=config.num_mem_tokens)
        self.rmt = RecurrentWrapperNoSegmentationGenerate(
            memory_cell,
            max_n_segments=config.max_n_segments,
            think_token_id=config.think_token_id,
            answer_token_id=config.answer_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id
        )

    def forward(self, labels=None, *args, **kwargs):
        return self.rmt(labels=labels, *args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.rmt.generate(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            print("Failed to load state, retrying with RMT loader.")
            self.rmt.load_state_dict(state_dict, strict=True, assign=assign)
            print("Success!")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, *args, **kwargs):
        from transformers.utils.hub import cached_file, HfHubHTTPError
        import torch

        if config is None:
            config = RMTConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config)

        state_dict = None
        try:
            weights_path = cached_file(pretrained_model_name_or_path, "model.safetensors", **kwargs)
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        except (OSError, HfHubHTTPError):
            try:
                weights_path = cached_file(pretrained_model_name_or_path, "pytorch_model.bin", **kwargs)
                state_dict = torch.load(weights_path, map_location="cpu")
            except (OSError, HfHubHTTPError):
                print(f"Warning: Could not find weights for {pretrained_model_name_or_path}. "
                      f"The model is initialized randomly.")

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)

        return model


class RMTForReasoningDynamicChunking(PreTrainedModel):
    """`RMTForReasoning` variant that swaps the inner wrapper to
    :class:`RecurrentWrapperDynamicChunking`.

    Why a separate class
    --------------------
    The original :class:`RMTForReasoning` is hard-wired to
    :class:`RecurrentWrapperNoSegmentationGenerate` and expects the dataloader
    to feed pre-split segments (``forward(segments=..., labels=...)``).
    Dynamic chunking is the *opposite* contract: the dataloader feeds a flat
    sequence and the chunker decides where to split. Rather than refactor the
    existing class we add a sibling with the flat-sequence forward signature
    expected by HuggingFace ``Trainer`` (``input_ids``, ``attention_mask``,
    ``labels``, ``labels_mask``).

    Reads extra chunker fields off the config (see ``RMTConfig``):
        * ``chunker_compression_ratio`` — target ``N`` for the H-Net ratio loss.
        * ``chunker_aux_loss_weight``   — λ in ``loss = lm_loss + λ · ratio_loss``.
        * ``chunker_ratio_loss_exclude_query_start`` — drop forced ``?`` from ratio loss.
        * ``k2``                         — truncated-BPTT window, forwarded to
          ``manage_gradients``.
    """
    config_class = RMTConfig

    def __init__(self, config: RMTConfig, **kwargs):
        super().__init__(config, **kwargs)
        from transformers import AutoConfig, AutoModelForCausalLM

        # Build the base causal LM the same way RMTForReasoning does.
        if config.from_pretrained:
            base_model = AutoModelForCausalLM.from_pretrained(config.from_pretrained)
        else:
            base_config = config.base_model_config \
                or AutoConfig.from_pretrained(config.base_model_name)
            base_model = AutoModelForCausalLM.from_config(base_config)

        self.rmt_config = config
        memory_cell = MemoryCell(base_model, num_mem_tokens=config.num_mem_tokens)
        self.rmt = RecurrentWrapperDynamicChunking(
            memory_cell,
            chunker_compression_ratio=getattr(config, 'chunker_compression_ratio', None),
            chunker_aux_loss_weight=getattr(config, 'chunker_aux_loss_weight', 0.01),
            max_n_segments=config.max_n_segments,
            k2=getattr(config, 'k2', -1),
            query_token_id=getattr(config, 'query_token_id', None),
            chunker_ratio_loss_exclude_query_start=getattr(
                config, 'chunker_ratio_loss_exclude_query_start', True,
            ),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, labels_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None,
                **kwargs):
        return self.rmt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            labels_mask=labels_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

    def generate(self, input_ids=None, attention_mask=None, **generate_kwargs):
        return self.rmt.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Same forgiving fallback as RMTForReasoning: try the standard
        # nn.Module loader first, fall back to the inner wrapper's state
        # dict if keys disagree (e.g. when a checkpoint omits the new
        # routing module).
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            print("Failed to load state, retrying with RMT loader.")
            self.rmt.load_state_dict(state_dict, strict=True, assign=assign)
            print("Success!")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, *args, **kwargs):
        # Mirrors RMTForReasoning.from_pretrained exactly; the only reason
        # to override it is so `cls(config)` instantiates *this* subclass.
        from transformers.utils.hub import cached_file, HfHubHTTPError
        import torch

        if config is None:
            config = RMTConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        model = cls(config)

        state_dict = None
        try:
            weights_path = cached_file(pretrained_model_name_or_path, "model.safetensors", **kwargs)
            from safetensors.torch import load_file
            state_dict = load_file(weights_path, device="cpu")
        except (OSError, HfHubHTTPError):
            try:
                weights_path = cached_file(pretrained_model_name_or_path, "pytorch_model.bin", **kwargs)
                state_dict = torch.load(weights_path, map_location="cpu")
            except (OSError, HfHubHTTPError):
                print(f"Warning: Could not find weights for {pretrained_model_name_or_path}. "
                      f"The model is initialized randomly.")

        if state_dict is not None:
            model.load_state_dict(state_dict, strict=False)

        return model


class MemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens):
        super().__init__()
        self.model = base_model
        self.create_memory(num_mem_tokens)

    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim = getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        self.read_memory_position = range(num_mem_tokens)
        self.write_memory_position = range(-num_mem_tokens, 0)

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def forward(self, input_ids, memory_state=None, **kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, write_mem=True, **kwargs)
        out = self.model(**seg_kwargs)
        out, new_memory_state = self.process_output(out, **kwargs)

        return out, new_memory_state

    def generate(self, input_ids, memory_state, attention_mask=None, **generate_kwargs):
        if memory_state is None:
            memory_state = self.set_memory(input_ids.shape)

        seg_kwargs = self.process_input(input_ids, memory_state, attention_mask=attention_mask, write_mem=False)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'],
                                  attention_mask=seg_kwargs['attention_mask'],
                                  **generate_kwargs)
        return out

    def process_input(self, input_ids, memory_state, write_mem, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if self.num_mem_tokens > 0:
            if write_mem:
                inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
            else:
                inputs_embeds = torch.cat([memory_state, inputs_embeds], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs

    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, self.num_mem_tokens: self.num_mem_tokens + attention_mask.shape[1]] = attention_mask
            return mask

    def process_output(self, model_outputs, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            out['logits'] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]

            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens]
                                        for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            memory_state = None
            out = model_outputs

        return out, memory_state


class RecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None,
                output_attentions=None, output_hidden_states=None):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        for seg_num, segment in enumerate(segmented):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, labels=labels,
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        memory_state = None
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out, memory_state = self.memory_cell(**segment, memory_state=memory_state, output_hidden_states=True)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, memory_state=memory_state, **generate_kwargs)

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments

    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1)
                                    for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))

            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]

            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        return out

    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
                or seg_num + k2 > max_n_segments:
            return memory_state

        memory_state = memory_state.detach()
        return memory_state

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class RecurrentWrapperDynamicChunking(RecurrentWrapper):
    """RMT recurrent wrapper that learns *where* to split the input into segments.

    High-level idea
    ---------------
    Vanilla RMT cuts the sequence into fixed-size windows of ``segment_size``
    tokens and recurrently passes a short memory state between them. Here we
    let an H-Net-style :class:`RoutingModule` predict, per token position,
    whether that position should *start* a new segment. The resulting segments
    are variable-length and content-adaptive, but the recurrent loop, the
    memory tokens, and everything inside :class:`MemoryCell` are untouched.

    Shape glossary used in this class
    ---------------------------------
        B    batch size
        L    padded sequence length (same across the batch — the dataloader
             already right-pads to the longest example in the batch)
        D    hidden / embedding dimension of the base model
        V    vocab size of the base model (only appears in ``logits``)
        K    number of segments. **K is variable per batch element**;
             ``max_K`` is the largest K in the current batch (optionally
             capped by ``rmt_config['max_n_segments']``).
        L_k  number of tokens in the k-th segment; also variable per batch
             element. We right-pad to ``max(L_k over batch)`` per segment.

    End-to-end data flow for one forward pass
    -----------------------------------------
        input_ids:        [B, L]   (long)
            │
            ▼  word embedding
        inputs_embeds:    [B, L, D]
            │
            ▼  RoutingModule (cos-sim between consecutive token embeddings)
        boundary_mask:    [B, L]   (bool, True where a new segment starts)
        boundary_prob:    [B, L, 2]
        selected_probs:   [B, L, 1]
            │
            ▼  STE gate (no-op in forward value, carries gradient backward)
        inputs_embeds:    [B, L, D]   (mathematically unchanged forward)
            │
            ▼  segment_by_boundaries
        segments = [
            { input_ids: [B, L_0], inputs_embeds: [B, L_0, D],
              attention_mask: [B, L_0], labels: [B, L_0], labels_mask: ... },
            { ... segment 1 ... },
            ...
            { ... segment max_K-1 ... },
        ]
            │
            ▼  recurrent MemoryCell forward over K segments, threading memory_state
        cell_outputs[k].logits:        [B, L_k, V]
            │
            ▼  process_outputs (per-segment masked CE, averaged over segments)
        out['loss']:   scalar
        out['logits']: [B, sum_k L_k, V]   (concatenated; padded within each L_k)

    Gradient flow to the chunker
    ----------------------------
    ``boundary_mask`` comes from an ``argmax`` (line 86 of
    ``chunker/dynamic_chunker.py``), so slicing-by-boundary is non-
    differentiable on its own. Two H-Net mechanisms are wired in to give the
    routing module a learning signal:

    1. **Straight-through (STE) gate**. We multiply ``inputs_embeds`` by a
       gate whose *forward value is 1.0* — so the memory cell sees unchanged
       embeddings — but whose backward path routes per-token gradient through
       ``selected_probs`` → ``boundary_prob`` → routing module weights. See
       :meth:`forward` for the construction.

    2. **Auxiliary ratio loss** (H-Net Algorithm 1). Pulls the average
       boundary probability ``G`` and the empirical boundary rate ``F`` toward
       ``1 / chunker_compression_ratio``. See :meth:`_compute_ratio_loss` for
       the derivation.

    Per-batch K varies
    ------------------
    Different batch elements can have different numbers of boundaries (and
    therefore different K). Within one segment *index* we right-pad to the
    longest entry in the batch and supply a per-segment ``attention_mask``.
    Batch elements that have already used up all their segments contribute an
    all-padding entry (their memory state is then updated by a degenerate
    forward over ``[mem, padding, mem]`` — effectively a near-identity step
    once attention masking kicks in).
    """

    def __init__(self, memory_cell, d_model=None,
                 chunker_compression_ratio=None,
                 chunker_aux_loss_weight=0.01,
                 chunker_ratio_loss_exclude_query_start=True,
                 query_token_id=None,
                 **rmt_kwargs):
        """
        Parameters
        ----------
        memory_cell : MemoryCell
            The same object that the fixed-segment wrapper takes. Wraps the
            base causal LM and prepends/appends memory tokens.
        d_model : int, optional
            Hidden dimension of the base model. If ``None`` we infer it from
            ``memory_cell.model.config.hidden_size`` (HF transformers) or
            ``.n_embd`` (GPT-2 style configs).
        chunker_compression_ratio : float or int, optional
            Target compression ``N`` for the routing module. The ratio loss
            attracts both ``F`` (empirical boundary rate) and ``G`` (mean
            boundary probability) toward ``1 / N``. ``N = 8`` means "aim for
            one segment per 8 input tokens".
            If ``None``, the ratio loss is never computed.
        chunker_aux_loss_weight : float
            Scalar λ added to the main LM loss as
            ``loss = lm_loss + λ * ratio_loss``. Set to 0 to turn the ratio
            loss off entirely. Default 0.01 follows the H-Net paper range
            scaled down for a small (~135M) backbone.
        chunker_ratio_loss_exclude_query_start : bool
            If True (default), positions of the forced query-start boundary
            (first ``?``) are removed from the ratio-loss ``valid`` mask, like
            position 0. Set False to count ``qs`` toward the global 1/N target.
        **rmt_kwargs
            Forwarded to :class:`RecurrentWrapper.__init__` and stored in
            ``self.rmt_config``. Notable keys we read elsewhere:
              * ``max_n_segments`` — cap on K and threshold used by
                :meth:`manage_gradients` to truncate BPTT.
              * ``k2`` — BPTT window; see ``RecurrentWrapper.manage_gradients``.
        """
        super().__init__(memory_cell, **rmt_kwargs)

        # Infer hidden size from the base model config if the caller didn't
        # pass it explicitly. We try both naming conventions (HF Llama-style
        # and GPT-2 style) to keep this wrapper drop-in for many bases.
        if d_model is None:
            base_config = memory_cell.model.config
            d_model = getattr(base_config, 'hidden_size', None) \
                or getattr(base_config, 'n_embd', None)
            if d_model is None:
                raise ValueError(
                    "Could not infer d_model from base model config; pass it explicitly."
                )

        # The routing module is the *only* new learnable component this
        # wrapper introduces. It owns two D×D projection matrices and predicts
        # one boundary score per token.
        self.routing_module = RoutingModule(d_model)
        self.chunker_compression_ratio = chunker_compression_ratio
        self.chunker_aux_loss_weight = chunker_aux_loss_weight
        self.chunker_ratio_loss_exclude_query_start = chunker_ratio_loss_exclude_query_start
        self.query_token_id = query_token_id

        # Defensive: if the user asks for a target compression ratio but
        # leaves the aux-loss weight at zero, the ratio loss is computed
        # (and exposed via out['ratio_loss']) but never contributes to
        # backward. Almost always a config mistake — warn loudly.
        if self.chunker_compression_ratio is not None and self.chunker_aux_loss_weight == 0.0:
            warnings.warn("chunker_compression_ratio is set but chunker_aux_loss_weight=0.0 — ratio loss will be computed but not applied.")
        if self.query_token_id is None:
            warnings.warn(
                "query_token_id is not set on RMTConfig — dynamic chunking will not "
                "isolate the KV query into its own segment."
            )

    @staticmethod
    def _infer_query_span(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_token_id: int,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Locate the query as the token span starting at the first ``?``.

        ``query_len`` runs from that ``?`` until the first supervised target
        token (``labels != -100``) when ``labels`` is given, otherwise until
        the end of the valid (non-pad) region.
        """
        batch_size, _ = input_ids.shape
        device = input_ids.device
        query_start = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        query_len = torch.ones((batch_size,), dtype=torch.long, device=device)

        for b in range(batch_size):
            valid_len = int(attention_mask[b].sum().item())
            if valid_len == 0:
                continue
            row_ids = input_ids[b, :valid_len]
            qmark_hits = (row_ids == query_token_id).nonzero(as_tuple=True)[0]
            if len(qmark_hits) == 0:
                continue
            qs = int(qmark_hits[0].item())
            query_start[b] = qs

            if labels is not None:
                row_labels = labels[b, :valid_len]
                target_hits = (row_labels != -100).nonzero(as_tuple=True)[0]
                q_end = int(target_hits[0].item()) if len(target_hits) > 0 else valid_len
            else:
                q_end = valid_len
            query_len[b] = max(q_end - qs, 1)

        return query_start, query_len

    def _apply_query_segment_boundaries(
        self,
        boundary_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Auto-detect ``?`` and force the query (and optional target) segments."""
        if self.query_token_id is None:
            return boundary_mask, None

        query_start, query_len = self._infer_query_span(
            input_ids, attention_mask, self.query_token_id, labels=labels,
        )
        if (query_start < 0).all():
            warnings.warn(
                "No '?' token found in batch — query segment enforcement skipped."
            )
            return boundary_mask, None

        boundary_mask = self._enforce_query_segment_boundaries(
            boundary_mask, attention_mask, query_start, query_len,
        )
        return boundary_mask, query_start

    @staticmethod
    def _enforce_query_segment_boundaries(
        boundary_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        query_start: torch.Tensor,
        query_len: torch.Tensor,
    ) -> torch.Tensor:
        """Force the query into its own segment; optionally isolate the target too.

        For each batch row ``b``:
          * ``boundary_mask[b, query_start[b]] = True`` — query begins its own segment.
          * No boundaries inside ``(query_start, query_start + query_len)`` so the
            query is never split.
          * If ``query_start + query_len < valid_len``, a boundary is forced at the
            first target token so the target can form a following segment (training).

        When the prompt is ``context + query`` only (no target tokens), the query
        segment is the **last** segment and is the right place to call ``generate``.
        """
        boundary_mask = boundary_mask.clone()
        valid = attention_mask.bool()
        batch_size, seq_len = boundary_mask.shape

        for b in range(batch_size):
            qs = int(query_start[b].item())
            ql = int(query_len[b].item())
            if qs < 0 or ql <= 0:
                continue
            valid_len = int(valid[b].sum().item())
            if qs >= valid_len:
                continue

            q_end = min(qs + ql, valid_len)
            # Strip spurious cuts inside the query span.
            if q_end > qs + 1:
                boundary_mask[b, qs + 1:q_end] = False
            boundary_mask[b, qs] = True

            # Target (if present) starts a new segment after the query.
            if q_end < valid_len:
                boundary_mask[b, q_end] = True

        boundary_mask = boundary_mask & valid
        boundary_mask[:, 0] = True
        return boundary_mask

    @staticmethod
    def _resolve_query_segment_index(segmented, query_start: torch.Tensor | None) -> int:
        """Index of the query-only segment in ``segmented`` (defaults to last)."""
        if query_start is None or len(segmented) == 0:
            return len(segmented) - 1
        batch_size = query_start.shape[0]
        indices = []
        for b in range(batch_size):
            qs = int(query_start[b].item())
            found = len(segmented) - 1
            for k, segment in enumerate(segmented):
                if int(segment['_slice_starts'][b].item()) == qs:
                    found = k
                    break
            indices.append(found)
        if len(set(indices)) != 1:
            warnings.warn(
                "Query segment index differs across batch rows; using the last "
                f"index ({indices[-1]}). Use batch size 1 for strict decode alignment."
            )
        return indices[-1]

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None,
                attention_mask=None, output_attentions=None, output_hidden_states=None):
        """One training-style forward pass.

        Expected shapes
        ---------------
            input_ids       [B, L]      long   (or None if inputs_embeds given)
            labels          [B, L]      long   (use -100 to ignore positions)
            labels_mask     [B, L]      bool/long, optional
            inputs_embeds   [B, L, D]   float, optional
            attention_mask  [B, L]      long, 1 = real token, 0 = padding

        Returns a HuggingFace ModelOutput with keys
            loss            scalar = lm_loss + λ * ratio_loss
            lm_loss         scalar, per-segment masked CE averaged over segments
            ratio_loss      scalar or None
            logits          [B, sum_k L_k, V]
            boundary_prob   [B, L, 2]   (p(no_boundary), p(boundary)) per token
            boundary_mask   [B, L]      bool, True at segment starts
        """
        # ---------------- step 1: ensure we have embeddings and a mask ----------------
        # [B, L, D].
        if inputs_embeds is None:
            inputs_embeds = self.memory_cell.model.get_input_embeddings()(input_ids)

        # If no attention mask was provided assume every position is real.
        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device,
            )

        # ---------------- step 2: predict boundaries ----------------------------------
        # RoutingModule computes cos-similarity between consecutive token
        # embeddings: low similarity → probable boundary. By construction it
        # forces position 0 to be a boundary with probability 1.0, so every
        # non-empty sequence yields at least one segment.
        #
        # Output shapes:
        #   routing_out.boundary_prob   [B, L, 2]   (p(no_b), p(b))
        #   routing_out.boundary_mask   [B, L]      bool, argmax over dim=-1
        #   routing_out.selected_probs  [B, L, 1]   p of the *chosen* class at each pos
        routing_out = self.routing_module(inputs_embeds, mask=attention_mask.bool())

        boundary_mask, query_start = self._apply_query_segment_boundaries(
            routing_out.boundary_mask,
            attention_mask,
            input_ids,
            labels=labels,
        )
        routing_out.boundary_mask = boundary_mask

        # ---------------- step 3: straight-through gate (STE) -------------------------
        # The discrete boundary_mask carries no gradient, so the routing
        # module's weights would never update from the LM loss without help.
        # The STE trick below has:
        #   forward  value = 1.0 (because (p - p.detach()) is numerically 0)
        #   backward grad  = d/d(selected_probs) of the multiplicative gate
        # Multiplying `inputs_embeds` by this 1.0-valued gate therefore leaves
        # the forward pass exactly unchanged (the memory cell sees the same
        # embeddings it would have seen otherwise), but creates a gradient
        # highway: dL/d(inputs_embeds[b, i]) ⟶ dL/d(selected_probs[b, i])
        # ⟶ dL/d(boundary_prob[b, i]) ⟶ dL/d(routing weights).
        #
        # We only build the gate when a graph is being constructed; otherwise
        # we skip the multiply because the result would be a numerical no-op
        # at non-trivial compute cost.
        if self.training and torch.is_grad_enabled():
            selected_probs = routing_out.selected_probs                            # [B, L, 1]
            ste_gate = 1.0 + (selected_probs - selected_probs.detach())            # [B, L, 1]
            # Broadcasting [B, L, 1] over [B, L, D] gives [B, L, D] back.
            inputs_embeds = inputs_embeds * ste_gate

        # ---------------- step 4: split inputs into variable-length segments ----------
        # `segmented` is a Python list of dicts of length max_K. Each dict
        # holds tensors that have been right-padded to the longest segment
        # within that index across the batch. See `segment_by_boundaries` for
        # the exact algorithm.
        segmented = self.segment_by_boundaries(
            boundary_mask=boundary_mask,
            attention_mask=attention_mask,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            labels_mask=labels_mask,
        )

        # ---------------- step 5: recurrent RMT forward over segments -----------------
        memory_state = None                       # [B, num_mem_tokens, D] after first call
        cell_outputs = []                         # one entry per segment
        for seg_num, segment in enumerate(segmented):
            # The memory cell prepends and appends memory tokens, runs the
            # base model, and returns the *next* memory state.
            cell_out, memory_state = self.memory_cell(
                input_ids=segment.get('input_ids'),
                inputs_embeds=segment.get('inputs_embeds'),
                attention_mask=segment.get('attention_mask'),
                memory_state=memory_state,
                output_hidden_states=True,
            )
            cell_outputs.append(cell_out)
            # Standard truncated-BPTT trick from the parent class: optionally
            # `detach` memory_state every k2 segments to bound the backward
            # graph depth. With k2 in {-1, None} the memory state is kept
            # attached for the full sequence.
            memory_state = self.manage_gradients(memory_state, seg_num)

        # ---------------- step 6: aggregate losses + per-segment outputs --------------
        # process_outputs returns the LM cross-entropy loss averaged across
        # segments (with all three of attention_mask, labels_mask and the
        # ignore_index=-100 convention respected). We pass the original L
        # so that `out['logits']` is scattered back to a shape-deterministic
        # [B, L, V] tensor (necessary for DataParallel gather).
        out = self.process_outputs(
            cell_outputs, segmented,
            original_seq_len=int(inputs_embeds.shape[1]),
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        main_loss = out.get('loss')
        ratio_loss = self._compute_ratio_loss(
            boundary_prob=routing_out.boundary_prob,
            boundary_mask=boundary_mask,
            attention_mask=attention_mask,
            query_start=query_start,
        )
        out['lm_loss'] = main_loss
        out['ratio_loss'] = ratio_loss


        if ratio_loss is not None and self.chunker_aux_loss_weight > 0 \
                and isinstance(main_loss, torch.Tensor):
            out['loss'] = main_loss + self.chunker_aux_loss_weight * ratio_loss

        out['boundary_prob'] = routing_out.boundary_prob
        out['boundary_mask'] = boundary_mask
        return out

    def generate(self, input_ids, attention_mask=None, inputs_embeds=None, **generate_kwargs):
        """Autoregressive generation with dynamic chunking on the context.

        Pass ``context + query`` token IDs (no answer/target). The first ``?``
        (``config.query_token_id``) starts the query segment; earlier segments
        only warm up memory, then ``memory_cell.generate`` decodes from the
        query segment.

        Shapes
        ------
            input_ids:        [B, L]
            return value:     whatever ``memory_cell.generate`` returns
                              (typically [B, L_gen] of token IDs).
        """
        # Same prep as forward: materialize embeddings and a mask.
        if inputs_embeds is None:
            inputs_embeds = self.memory_cell.model.get_input_embeddings()(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(
                inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device,
            )

        # Predict boundaries on the context exactly as in forward.
        routing_out = self.routing_module(inputs_embeds, mask=attention_mask.bool())

        boundary_mask, query_start = self._apply_query_segment_boundaries(
            routing_out.boundary_mask,
            attention_mask,
            input_ids,
            labels=None,
        )

        # STE has forward value 1.0 (mathematical no-op). At pure inference
        # time (``torch.no_grad()``) we skip the multiply because there's no
        # graph to populate. If `generate` is ever called inside a training
        # loop with grads enabled the STE still builds correctly.
        if torch.is_grad_enabled():
            selected_probs = routing_out.selected_probs                            # [B, L, 1]
            ste_gate = 1.0 + (selected_probs - selected_probs.detach())
            inputs_embeds = inputs_embeds * ste_gate

        # Same segmentation as forward, except we don't have labels.
        segmented = self.segment_by_boundaries(
            boundary_mask=boundary_mask,
            attention_mask=attention_mask,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )

        # `segmented` is empty only if the entire context was padding — a
        # caller bug. Surface it loudly instead of returning garbage.
        if not segmented:
            raise ValueError("Dynamic chunker produced zero segments; nothing to generate from.")

        query_seg_idx = self._resolve_query_segment_index(segmented, query_start)

        # Warm up every segment strictly before the query segment.
        memory_state = None
        for segment in segmented[:query_seg_idx]:
            _, memory_state = self.memory_cell(
                input_ids=segment.get('input_ids'),
                inputs_embeds=segment.get('inputs_embeds'),
                attention_mask=segment.get('attention_mask'),
                memory_state=memory_state,
                output_hidden_states=True,
            )

        # Decode from the query segment (must be context+query input, no target).
        final_segment = segmented[query_seg_idx]
        out = self.memory_cell.generate(
            input_ids=final_segment.get('input_ids'),
            inputs_embeds=final_segment.get('inputs_embeds'),
            attention_mask=final_segment.get('attention_mask'),
            memory_state=memory_state,
            **generate_kwargs,
        )
        return out

    # TODO: vectorise this loop — currently O(B × max_n_segments) Python iterations
    def segment_by_boundaries(self, boundary_mask, attention_mask, **tensors):
        """Group tokens into variable-length segments delimited by ``boundary_mask``.

        Semantics
        ---------
        ``boundary_mask[b, i] == True`` means position ``i`` is the *start* of
        a new segment in batch element ``b``. By construction (see
        :class:`RoutingModule`) position 0 is always a boundary, so every
        non-empty sequence yields at least one segment.

        Concrete example (B=2, L=8)
        ---------------------------
            attention_mask  = [[1,1,1,1,1,1,1,1],     # 8 real tokens
                               [1,1,1,1,1,1,0,0]]     # 6 real, 2 padding
            boundary_mask   = [[T,F,F,T,F,T,F,F],     # boundaries at 0,3,5
                               [T,F,T,F,F,T,F,F]]     # boundaries at 0,2,5

        After this method::

            segmented[0] = {                          # first segment of each batch elem
                'input_ids':      [[t0,t1,t2], [t0,t1,0]],  # right-pad to max len 3
                'attention_mask': [[1,1,1],    [1,1,0]],
            }
            segmented[1] = {                          # second segment
                'input_ids':      [[t3,t4,0],  [t2,t3,t4]],
                'attention_mask': [[1,1,0],    [1,1,1]],
            }
            segmented[2] = {                          # third segment
                'input_ids':      [[t5,t6,t7], [t5,0,0]],
                'attention_mask': [[1,1,1],    [1,0,0]],
            }

        Both batch elements happen to have K=3 here, but if one had fewer
        boundaries the missing slots would be all-padding (``attention_mask``
        all zeros), and the MemoryCell would still produce a valid (but
        meaningless) update for them.

        Parameters
        ----------
        boundary_mask : torch.BoolTensor, shape ``[B, L]``
        attention_mask : torch.LongTensor, shape ``[B, L]``
        **tensors : per-key tensors of shape ``[B, L, ...]`` to be sliced in
            lockstep with the segmentation (e.g. ``input_ids``,
            ``inputs_embeds``, ``labels``, ``labels_mask``). ``None`` values
            are passed through.

        Returns
        -------
        list of dict
            Length is ``max_K`` (largest K across the batch, optionally capped
            by ``rmt_config['max_n_segments']``). Each dict has the same keys
            as ``tensors`` plus an ``attention_mask`` aligned to that
            segment's right-padded length.
        """
        # Make sure boundaries inside padded regions don't count: a boundary
        # at a padding position would otherwise create an empty/invalid
        # segment further down.
        valid_mask = attention_mask.bool()                              # [B, L] bool
        boundary_mask = boundary_mask & valid_mask                      # [B, L] bool
        batch_size, _ = boundary_mask.shape
        device = boundary_mask.device

        # ---- pass 1: for each batch element, materialise the list of
        # (start, end) slice indices that define its segments.
        # We loop in Python because K differs per batch element, which makes
        # this hard to vectorise without padding to a worst-case K.
        per_batch_ranges = []                                           # list of list[(s, e)]
        for b in range(batch_size):
            # Indices of every True in boundary_mask[b]. Always includes 0
            # thanks to RoutingModule's forced first boundary.
            positions = boundary_mask[b].nonzero(as_tuple=True)[0].tolist()
            # number of non-padding tokens for this batch element 
            # == implicit end of the last segment.
            valid_len = int(valid_mask[b].sum().item())
            ranges = []
            for i, start in enumerate(positions):
                # Segment k spans [positions[k], positions[k+1]); the final
                # segment runs until the end of the *valid* region.
                end = positions[i + 1] if i + 1 < len(positions) else valid_len
                if end > start:                                          # skip degenerate empty ranges
                    ranges.append((start, end))
            per_batch_ranges.append(ranges)

        # max_K across the batch (with optional cap so very long sequences
        # don't blow up the BPTT graph).
        max_n_segments = max((len(r) for r in per_batch_ranges), default=0)
        cap = self.rmt_config.get('max_n_segments')
        if cap is not None and cap > 0:
            max_n_segments = min(max_n_segments, cap)


        if max_n_segments == 0:
            return []

        # ---- pass 2: build one batched dict per segment index.
        segmented = []
        for seg_idx in range(max_n_segments):
            # Collect this segment's (start, end) for every batch element.
            # Batch elements with fewer than `seg_idx+1` segments get an
            # all-zero entry that becomes an all-padding slot below.
            starts, ends, seg_lens = [], [], []
            for b in range(batch_size):
                if seg_idx < len(per_batch_ranges[b]):
                    s, e = per_batch_ranges[b][seg_idx]
                else:
                    s, e = 0, 0
                starts.append(s)
                ends.append(e)
                seg_lens.append(e - s)

            # Right-pad to the longest segment in this slot. The `or 1`
            # avoids constructing a zero-length tensor when *every* batch
            # element happens to be exhausted (rare, but possible if the cap
            # kicks in unevenly).
            max_seg_len = max(seg_lens) if max(seg_lens) > 0 else 1

            # Slice each tensor in `tensors` to this segment and right-pad
            # to max_seg_len. Output shape is [B, max_seg_len, *tail].
            segment_dict = {}
            for key, tensor in tensors.items():
                if tensor is None:
                    segment_dict[key] = None
                    continue
                # Build the padded buffer with the input dtype, in zeros so
                # any padded position is safe to feed through embeddings /
                # losses (and the attention mask will zero out anyway).
                out_shape = [batch_size, max_seg_len] + list(tensor.shape[2:])
                padded = torch.zeros(out_shape, dtype=tensor.dtype, device=tensor.device)
                for b in range(batch_size):
                    if seg_lens[b] > 0:
                        # Copy this batch element's slice into the front of
                        # the padded buffer; positions [seg_lens[b]:] stay 0.
                        padded[b, :seg_lens[b]] = tensor[b, starts[b]:ends[b]]
                segment_dict[key] = padded

            # Build the per-segment attention mask matching the right-pad
            # layout: 1 for real tokens of *this* segment, 0 for padding.
            seg_attn = torch.zeros(
                batch_size, max_seg_len,
                dtype=attention_mask.dtype, device=device,
            )
            for b in range(batch_size):
                if seg_lens[b] > 0:
                    seg_attn[b, :seg_lens[b]] = 1
            segment_dict['attention_mask'] = seg_attn

            # Slice metadata. Used by `process_outputs` to scatter the
            # per-segment logits back into a `[B, L, V]` tensor that has
            # the same shape on every DataParallel replica (otherwise the
            # gather at the end of DP fails — see the note in
            # `process_outputs`).
            segment_dict['_slice_starts'] = torch.tensor(starts, dtype=torch.long, device=device)
            segment_dict['_slice_ends']   = torch.tensor(ends,   dtype=torch.long, device=device)
            segmented.append(segment_dict)

        return segmented

    def process_outputs(self, cell_outputs, segments, **kwargs):
        """Aggregate loss over dynamically-sized segments with proper masking.

        Why per-segment loss?
        ---------------------
        In fixed-segment RMT every batch element has the same segment
        boundaries, so concatenating per-segment logits gives a tensor whose
        positions line up with the original label tensor and a single CE
        over the whole thing is correct. With *dynamic* chunking different
        batch elements have different boundaries, so the concatenation
        interleaves real positions with padding and the labels no longer
        align position-for-position. Per-segment loss with per-segment masks
        avoids that mess.

        Masking
        -------
        Per segment we intersect three masks before computing CE:
          1. ``attention_mask``  — drop the right-pad introduced by
             cross-batch length alignment within this segment.
          2. ``labels_mask``     — (optional) drop non-target positions
             such as context tokens in a context/query setup.
          3. ``labels != -100``  — drop ignore-index positions. This is the
             convention used by ``collate_fn_dynamic``: context labels are
             set to -100, query labels are real token IDs. With dynamic
             chunking a segment can straddle the context/query boundary;
             this mask keeps only the query-side positions of such a
             "mixed" segment.

        Segments that end up with zero valid positions (e.g. an all-context
        or all-padding segment) are skipped entirely so we never feed an
        empty batch to ``CrossEntropyLoss`` (which would otherwise NaN).
        """
        out = CausalLMOutputWithCrossAttentions()

        # We collect one scalar CE loss per segment that contributed any
        # valid tokens, then average at the end.
        losses = []
        for cell_out, segment in zip(cell_outputs, segments):
            # full_logits[b, t, v] = score of token v at position t in this
            # segment for batch element b. Shape [B, L_k, V].
            full_logits = cell_out.logits
            labels = segment.get('labels')
            if labels is None:
                # Caller is running inference: no labels, nothing to score.
                continue

            # Teacher forcing shift: we predict token t from positions <t,
            # so logits[..., :-1, :] should be matched against labels[..., 1:].
            # Shapes after the shift:
            #   shift_labels  [B, L_k - 1]
            #   shift_logits  [B, L_k - 1, V]
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            # Flatten the (B, L_k - 1) grid into a single vector of length
            # B*(L_k - 1) so we can index out only the valid positions.
            flat_labels = shift_labels.view(-1)                                # [B*(L_k - 1)]
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))         # [B*(L_k - 1), V]

            # Build a [B, L_k - 1] boolean mask that keeps only positions we
            # want to score. Each source mask is shifted by 1 so it aligns
            # with shift_labels (the prediction at logit index t corresponds
            # to label at position t + 1).
            attn = segment.get('attention_mask')
            if attn is not None:
                valid = attn[..., 1:].contiguous().bool()                      # padding-aware
            else:
                valid = torch.ones_like(shift_labels, dtype=torch.bool)
            labels_mask = segment.get('labels_mask')
            if labels_mask is not None:
                valid = valid & labels_mask[..., 1:].contiguous().bool()       # user-provided
            valid = valid & (shift_labels != -100)                             # ignore_index

            flat_valid = valid.view(-1)                                        # [B*(L_k - 1)]
            if flat_valid.any():
                # CE over the surviving positions only. We do NOT pass
                # ignore_index here because we've already filtered.
                loss_fct = CrossEntropyLoss()
                losses.append(loss_fct(flat_logits[flat_valid], flat_labels[flat_valid]))

        # Reduce per-segment losses to a single scalar. Segments with no
        # valid positions are excluded from both numerator and denominator
        # so they don't artificially deflate the average.
        if losses:
            out['loss'] = sum(losses) / len(losses)
        elif cell_outputs:
            # Labels were provided but every segment was fully masked
            # (e.g. an eval sample with no query tokens). Return a zero on
            # the right device/dtype so the training loop doesn't crash on
            # .backward()/.item().
            ref = cell_outputs[0].logits
            out['loss'] = torch.zeros((), device=ref.device, dtype=ref.dtype)
        else:
            # No segments at all (degenerate input).
            out['loss'] = 0

        # Scatter per-segment logits back into a `[B, L, V]` tensor aligned
        # with the original input positions.
        #
        # Why scatter instead of concatenate?
        # -----------------------------------
        # Concatenating along dim=1 gives a `[B, sum_k L_k, V]` tensor whose
        # second dimension depends on the chunker's decisions. Under
        # ``nn.DataParallel`` each GPU runs the chunker on its own
        # mini-batch and ends up with *different* sum_k L_k. The final
        # ``DataParallel.gather`` then tries to concat-along-dim=0 tensors
        # whose dim=1 sizes disagree and crashes with
        # ``RuntimeError: Input tensor at index 1 has invalid shape ...``.
        #
        # Scattering back to `[B, L, V]` makes the output shape a function
        # of the inputs only (L is the same on every replica), so gather
        # works. It also makes downstream ``compute_metrics`` simpler
        # because logits now align position-for-position with labels.
        #
        # Caveat: the logit at the LAST position of each segment was *not*
        # used for the per-segment CE during training (the teacher-forcing
        # shift dropped it). When the metric function does a global
        # ``[..., :-1, :]`` shift it will now pair that logit with the
        # first label of the NEXT segment. The label at that position is
        # typically ignore-index (-100) in context-then-query setups, so
        # this rarely matters in practice — but it's worth flagging if you
        # use this to compute strict eval metrics.
        original_seq_len = kwargs.get('original_seq_len')
        if cell_outputs:
            ref = cell_outputs[0].logits                                    # [B, L_0, V]
            B = ref.shape[0]
            V = ref.shape[-1]
            L = original_seq_len if original_seq_len is not None else \
                int(max(int(seg['_slice_ends'].max().item()) for seg in segments))

            aligned_logits = torch.zeros(B, L, V, dtype=ref.dtype, device=ref.device)
            for cell_out, segment in zip(cell_outputs, segments):
                seg_logits = cell_out.logits                                # [B, max_seg_len, V]
                starts = segment['_slice_starts']                           # [B]
                ends   = segment['_slice_ends']                             # [B]
                for b in range(B):
                    s = int(starts[b]); e = int(ends[b])
                    if e > s:
                        aligned_logits[b, s:e] = seg_logits[b, :e - s]
            out['logits'] = aligned_logits                                  # [B, L, V]

            if kwargs.get('output_hidden_states') and getattr(cell_outputs[0], 'hidden_states', None) is not None:
                # Scatter hidden states with the same layout, one [B, L, D]
                # tensor per transformer layer.
                num_layers = len(cell_outputs[0].hidden_states)
                D = cell_outputs[0].hidden_states[0].shape[-1]
                aligned_hidden = [
                    torch.zeros(B, L, D,
                                dtype=cell_outputs[0].hidden_states[0].dtype,
                                device=ref.device)
                    for _ in range(num_layers)
                ]
                for cell_out, segment in zip(cell_outputs, segments):
                    starts = segment['_slice_starts']; ends = segment['_slice_ends']
                    for layer_idx in range(num_layers):
                        seg_hs = cell_out.hidden_states[layer_idx]          # [B, L_k, D]
                        for b in range(B):
                            s = int(starts[b]); e = int(ends[b])
                            if e > s:
                                aligned_hidden[layer_idx][b, s:e] = seg_hs[b, :e - s]
                out['hidden_states'] = tuple(aligned_hidden)

        return out

    def _compute_ratio_loss(
        self,
        boundary_prob,
        boundary_mask,
        attention_mask,
        query_start: torch.Tensor | None = None,
    ):
        """H-Net auxiliary ratio loss (Algorithm 1).

        Goal
        ----
        Push the chunker toward producing exactly ``1 / N`` boundaries per
        token, where ``N = chunker_compression_ratio``. Without this loss
        the chunker has two trivial degenerate solutions (everything is a
        boundary, or nothing is) that the LM loss alone cannot escape.

        Definitions (computed over *valid, learnable* positions only)
        --------------------------------------------------------------
          F = (1 / |valid|) * Σ_{i in valid} boundary_mask[i]
              = empirical fraction of positions that are boundaries.
              **detached** from the graph because boundary_mask comes from
              an argmax; no gradient flows through it anyway. F acts here as
              a constant *control signal* that tells the loss whether the
              chunker is currently over- or under-firing.
          G = (1 / |valid|) * Σ_{i in valid} p_boundary[i]
              = average predicted boundary probability. Differentiable.

        The H-Net formula
        -----------------
          L_ratio = (N / (N - 1)) * ((N - 1) · F · G + (1 - F) · (1 - G))

        Why this works (gradient view, F treated as constant)::
          ∂L_ratio / ∂G = (N / (N - 1)) · (N · F − 1)
        So:
          F > 1/N  (too many boundaries) → ∂L/∂G > 0 → push G down
          F < 1/N  (too few)             → ∂L/∂G < 0 → push G up
        Because F follows G through the argmax (more p ⇒ more positions
        crossing the 0.5 threshold), the system converges to F = G = 1/N.

        Excluded positions (not in ``valid``)
        -------------------------------------
        * Position 0 — ``RoutingModule`` forces a segment start there.
        * Query start ``qs`` (optional) — enforced by
          ``_enforce_query_segment_boundaries`` when
          ``chunker_ratio_loss_exclude_query_start`` is True.

        Parameters
        ----------
        boundary_prob   : [B, L, 2] — (p(no_boundary), p(boundary)) per pos.
        boundary_mask   : [B, L]    — hard argmax over boundary_prob.
        attention_mask  : [B, L]    — 1 for real tokens, 0 for padding.
        query_start     : [B] long, optional — per-row index of first ``?``;
            from :meth:`_infer_query_span`. Rows with ``query_start[b] < 0``
            are skipped.

        Returns
        -------
        Scalar tensor (mean over batch), or ``None`` if no compression
        target is configured.
        """
        N = self.chunker_compression_ratio
        if N is None or N <= 1:
            return None

        # valid[b, i] = 1.0 for learnable, non-padding positions; 0.0 for
        # padding and for structural boundaries excluded below.
        valid = attention_mask.bool().float()                                   # [B, L]
        valid[:, 0] = 0.0
        if (
            self.chunker_ratio_loss_exclude_query_start
            and query_start is not None
        ):
            batch_size, seq_len = valid.shape
            has_query = query_start >= 0
            if has_query.any():
                rows = torch.arange(batch_size, device=valid.device)[has_query]
                cols = query_start[has_query].clamp(max=seq_len - 1)
                valid[rows, cols] = 0.0
        # Clamp to 1 to avoid div-by-zero on degenerate batches where every
        # position is masked out.
        denom = valid.sum(dim=-1).clamp(min=1.0)                                # [B]

        # G — mean boundary probability over valid positions. Differentiable.
        # boundary_prob is [B, L, 2]; index 1 is p(boundary).
        p_boundary = boundary_prob[..., 1]                                      # [B, L]
        G = (p_boundary * valid).sum(dim=-1) / denom                            # [B]

        # F — empirical boundary rate over valid positions. Treated as a
        # constant signal (detach to be explicit even though argmax already
        # blocks the gradient). We `&` with attention_mask.bool() so any
        # stray boundaries inside padded regions don't get counted.
        F = ((boundary_mask & attention_mask.bool()).float() * valid).sum(dim=-1) / denom
        F = F.detach()                                                          # [B]

        # H-Net Algorithm 1.
        N = float(N)
        ratio = N / (N - 1.0)
        loss_per_batch = ratio * ((N - 1.0) * F * G + (1.0 - F) * (1.0 - G))    # [B]
        return loss_per_batch.mean()                                            # scalar


class RecurrentWrapperNoSegmentation(RecurrentWrapper):
    def forward(self, segments, labels, output_attentions=None, output_hidden_states=None):
        memory_state = None

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(input_ids=segment['input_ids'],
                                                      attention_mask=segment['attention_mask'],
                                                      memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            memory_state = self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, segments, **generate_kwargs):
        raise NotImplementedError("Generation not implemented for this wrapper.")

    def process_outputs(self, cell_outputs, segments, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        proxy_out = {}
        for seg_num, segment in enumerate(segments):
            cell_out = cell_outputs[seg_num]

            full_logits = cell_out.logits

            labels = segment.get('labels')
            if labels is not None:
                shift_labels = labels[..., 1:].contiguous()
                shift_logits = full_logits[..., :-1, :].contiguous()
                flat_labels = shift_labels.view(-1)
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))

                loss_fct = CrossEntropyLoss()
                labels_mask = segment.get('labels_mask')
                if labels_mask is not None:
                    shift_mask = labels_mask[..., :-1].contiguous()

                    flat_labels = flat_labels[shift_mask.view(-1)]
                    flat_logits = flat_logits[shift_mask.view(-1)]

                    if labels_mask.sum() == 0:
                        loss_value = 0
                    else:
                        loss_value = loss_fct(flat_logits, flat_labels)

                proxy_out[f'loss_{seg_num}'] = loss_value
            else:
                proxy_out[f'loss_{seg_num}'] = 0

            segment_keys = ['loss']
            if kwargs.get('output_attentions'):
                segment_keys.append('attentions')
            if kwargs.get('output_hidden_states'):
                segment_keys.append('hidden_states')

            for key, value in cell_out.items():
                if any([sk in key for sk in segment_keys]):
                    proxy_out[f'{key}_{seg_num}'] = value

        num_segments = len(segments)
        out['loss'] = sum([proxy_out[f'loss_{seg_num}'] for seg_num in range(num_segments)]) / num_segments
        out['logits'] = torch.cat([cell_out.logits for cell_out in cell_outputs], dim=1)
        # print(out.keys(), out.loss)

        return out

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            return self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class StopOnSpecialTokenCriteria(StoppingCriteria):
    def __init__(self, special_token_ids):
        self.special_token_ids = set(special_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        last_token = input_ids[0, -1].item()
        return last_token in self.special_token_ids


class RecurrentWrapperNoSegmentationGenerate(RecurrentWrapperNoSegmentation):
    def forward(self, segments, labels, output_attentions=None, output_hidden_states=None, *args, **kwargs):
        memory_state = None

        cell_outputs = []
        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(input_ids=segment['input_ids'],
                                                      attention_mask=segment['attention_mask'],
                                                      memory_state=memory_state, output_hidden_states=True)
            cell_outputs.append(cell_out)
            self.manage_gradients(memory_state, seg_num)

        out = self.process_outputs(cell_outputs, segments,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states)
        return out

    def generate(self, segments, **kwargs):
        memory_state = None

        for seg_num, segment in enumerate(segments):
            cell_out, memory_state = self.memory_cell(input_ids=segment['input_ids'],
                                                      attention_mask=segment['attention_mask'],
                                                      memory_state=memory_state, output_hidden_states=True)

        generated_segments = []
        for seg_num in range(len(segments), self.rmt_config.get("max_n_segments", 32)):
            output_ids, memory_state = self.generate_segment(memory_state=memory_state, **kwargs)
            generated_segments.append(output_ids)

            if self.all_done(generated_segments):
                break

        return generated_segments

    def generate_segment(self, memory_state, **kwargs):
        input_ids = self.get_bos_tensor(memory_state)
        attention_mask = torch.ones_like(input_ids).bool()

        generated = self.memory_cell.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            memory_state=memory_state,
            stopping_criteria=self.make_custom_stopping_criteria(),
            **kwargs
        )

        # Update memory state from generation
        fwd_inputs = torch.cat((input_ids, generated), dim=1)[:, :-1]
        _, memory_state = self.memory_cell(input_ids=fwd_inputs, memory_state=memory_state)

        return generated, memory_state

    def get_bos_tensor(self, memory_state):
        bos = self.rmt_config["bos_token_id"]
        bos_tensor = torch.tensor([bos] * memory_state.shape[0]).reshape(-1, 1)
        return bos_tensor.to(memory_state.device)

    def all_done(self, generated_segments):
        eos = self.rmt_config['eos_token_id']
        bs = generated_segments[0].shape[0]
        have_eos = [any([eos in seg[i] for seg in generated_segments]) for i in range(bs)]
        all_done = all(have_eos)
        return all_done

    def make_custom_stopping_criteria(self):
        return [StopOnSpecialTokenCriteria([self.rmt_config['think_token_id'], self.rmt_config['answer_token_id']])]
