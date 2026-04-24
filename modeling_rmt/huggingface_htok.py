"""
HD-RMT Stage 1: Adaptive Segmentation for RMT.

New classes only. Existing classes in huggingface.py are untouched.
The single integration point: AdaptiveRecurrentWrapper replaces the
fixed-stride segmentation loop with DynamicChunker-based pooling.

Architecture per context segment (vs. fixed RMT):
  Fixed:    [B, L_seg, D] → full self-attention over L_seg tokens + memory
  Adaptive: [B, L, D] → DynamicChunker → [B, K, D]
            each of K chunks [B, 1, D] processed recurrently with memory tokens
"""

from typing import Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .huggingface import MemoryCell, RMTConfig


class AdaptiveRMTConfig(RMTConfig):
    model_type = "adaptive_rmt"

    def __init__(
        self,
        use_adaptive_chunking: bool = True,
        n_chunks: int = 4,
        chunker_hidden_dim: Optional[int] = None,
        chunker_sigma: float = 1.0,
        chunker_encoder_type: str = "conv",
        hard_inference: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_adaptive_chunking = use_adaptive_chunking
        self.n_chunks = n_chunks
        self.chunker_hidden_dim = chunker_hidden_dim
        self.chunker_sigma = chunker_sigma
        self.chunker_encoder_type = chunker_encoder_type
        self.hard_inference = hard_inference


class AdaptiveRecurrentWrapper(torch.nn.Module):
    """
    Replaces fixed-stride segmentation with DynamicChunker pooling.

    Forward pass:
      1. Concatenate all context segment input_ids → embed → [B, L, D]
      2. DynamicChunker → [B, K, D] pooled chunks + boundary_scores [B, L]
      3. Process each chunk [B, 1, D] through MemoryCell recurrently
      4. Process query+target segment normally (token-level, same as original RMT)
      5. Loss only on query+target segment

    The memory recurrence mechanism is identical to original RMT.
    Only the "what goes into each recurrent step" changes.
    """

    def __init__(
        self,
        memory_cell: MemoryCell,
        chunker: torch.nn.Module,
        n_chunks: int,
        hard_inference: bool = False,
        **rmt_kwargs,
    ):
        super().__init__()
        self.memory_cell = memory_cell
        self.chunker = chunker
        self.n_chunks = n_chunks
        self.hard_inference = hard_inference
        self.rmt_config = rmt_kwargs

    def forward(
        self,
        segments,
        labels,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs,
    ):
        """
        Args:
            segments: list of dicts with input_ids, attention_mask, labels, labels_mask.
                      Last segment is query+target; all others are context.
            labels:   [B, total_L] — full concatenated labels (used by Trainer for logging)
        """
        context_segments = segments[:-1]
        query_segment = segments[-1]

        memory_state = None

        if context_segments:
            context_ids = torch.cat(
                [s["input_ids"] for s in context_segments], dim=1
            )  # [B, L]
            B = context_ids.shape[0]

            # Initialize memory before the chunk loop (cannot rely on MemoryCell's
            # internal init when input_ids=None for chunk steps).
            memory_state = self.memory_cell.set_memory((B, 1))

            embeddings = self.memory_cell.model.get_input_embeddings()(
                context_ids
            )  # [B, L, D]

            use_hard = self.hard_inference and not self.training
            chunks, _boundary_scores = self.chunker(embeddings, hard=use_hard)
            # chunks: [B, K, D]

            for k in range(self.n_chunks):
                chunk_embed = chunks[:, k : k + 1, :]  # [B, 1, D]
                _, memory_state = self.memory_cell(
                    input_ids=None,
                    inputs_embeds=chunk_embed,
                    memory_state=memory_state,
                    output_hidden_states=True,
                )
                memory_state = self._manage_gradients(memory_state, k)

        # Query+target: processed token-level (same as original RMT).
        # memory_state=None here means MemoryCell inits fresh from input shape.
        query_out, _ = self.memory_cell(
            input_ids=query_segment["input_ids"],
            attention_mask=query_segment["attention_mask"],
            memory_state=memory_state,
            output_hidden_states=True,
        )

        return self._compute_loss(query_out, query_segment)

    def _compute_loss(
        self,
        cell_out: CausalLMOutputWithCrossAttentions,
        segment: dict,
    ) -> CausalLMOutputWithCrossAttentions:
        out = CausalLMOutputWithCrossAttentions()
        logits = cell_out.logits  # [B, L_query, V]

        seg_labels = segment.get("labels")
        labels_mask = segment.get("labels_mask")

        if (
            seg_labels is not None
            and labels_mask is not None
            and labels_mask.sum() > 0
        ):
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = seg_labels[..., 1:].contiguous()
            shift_mask = labels_mask[..., :-1].contiguous()

            flat_logits = shift_logits.view(-1, shift_logits.size(-1))[
                shift_mask.view(-1)
            ]
            flat_labels = shift_labels.view(-1)[shift_mask.view(-1)]

            out["loss"] = CrossEntropyLoss()(flat_logits, flat_labels)
        else:
            # Zero loss that keeps the graph alive for gradient checks.
            out["loss"] = logits.sum() * 0.0

        out["logits"] = logits
        return out

    def _manage_gradients(
        self, memory_state: torch.Tensor, chunk_idx: int
    ) -> torch.Tensor:
        k2 = self.rmt_config.get("k2")
        max_n = self.rmt_config.get("max_n_segments", self.n_chunks + 1)
        if chunk_idx == 0 or k2 in {-1, None} or chunk_idx + k2 > max_n:
            return memory_state
        return memory_state.detach()

    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.memory_cell.model, "gradient_checkpointing_enable"):
            self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)


class RMTForAdaptiveReasoning(PreTrainedModel):
    """
    Top-level model: base LLM + MemoryCell + AdaptiveRecurrentWrapper + DynamicChunker.
    Drop-in replacement for RMTForReasoning when use_adaptive_chunking=True.
    """

    config_class = AdaptiveRMTConfig

    def __init__(self, config: AdaptiveRMTConfig, **kwargs):
        super().__init__(config, **kwargs)

        if config.from_pretrained:
            base_model = AutoModelForCausalLM.from_pretrained(config.from_pretrained)
        else:
            if config.base_model_config is None:
                base_config = AutoConfig.from_pretrained(config.base_model_name)
            else:
                base_config = config.base_model_config
            base_model = AutoModelForCausalLM.from_config(base_config)

        d_model = getattr(
            base_config, "n_embd", getattr(base_config, "hidden_size", None)
        )
        assert d_model is not None, (
            "Cannot infer d_model from base_model_config. "
            "Set n_embd or hidden_size in the config."
        )

        memory_cell = MemoryCell(base_model, num_mem_tokens=config.num_mem_tokens)

        from chunker.dynamic_chunker import DynamicChunker

        chunker = DynamicChunker(
            d_model=d_model,
            n_chunks=config.n_chunks,
            d_hidden=config.chunker_hidden_dim,
            sigma=config.chunker_sigma,
            encoder_type=config.chunker_encoder_type,
        )

        self.rmt = AdaptiveRecurrentWrapper(
            memory_cell=memory_cell,
            chunker=chunker,
            n_chunks=config.n_chunks,
            hard_inference=config.hard_inference,
            max_n_segments=config.max_n_segments,
        )

    def forward(self, labels=None, *args, **kwargs):
        return self.rmt(labels=labels, *args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        try:
            return super().load_state_dict(state_dict, strict, assign)
        except RuntimeError:
            print("Failed to load state, retrying with rmt sub-module loader.")
            self.rmt.load_state_dict(state_dict, strict=True, assign=assign)
            print("Success!")
