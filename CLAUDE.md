# HD-RMT Stage 1: Adaptive Segmentation for RMT

## Project goal

Replace fixed-size chunking in RMT with a learned dynamic chunker that produces
content-adaptive variable-length chunks. This is stage 1 of the HD-RMT project.
No byte-level tokenization yet — input is standard token embeddings. No hierarchical
stacking yet — single RMT level with one chunking layer before it.

The core hypothesis: content-adaptive chunk boundaries improve RMT memory utilization
because memory tokens aggregate semantically coherent units rather than arbitrary
fixed-size windows. Fixed chunking splits natural units (bracket pairs, edge
descriptions, reasoning steps) mid-boundary, diluting what each memory token
represents.

---

## Repo conventions

Before changing anything, read the existing RMT implementation completely. The
integration point is narrow: the chunker replaces the fixed striding that partitions
the input sequence into segments before each RMT forward pass. Everything else —
memory token initialization, cross-segment recurrence, loss computation — stays
unchanged.

Do not refactor existing RMT code. Add new files, import into existing ones at
the single integration point only.

---

## Architecture: DynamicChunker

### What it does

Takes token embeddings `x` of shape `[B, L, D]` and produces:
- `chunks`: tensor of shape `[B, K, D]` — K chunk representations, each D-dimensional
- `boundary_scores`: tensor of shape `[B, L]` — per-position boundary probability,
  used for analysis and visualization, not directly supervised

K is a hyperparameter (target number of chunks). At training time, K replaces
`L // chunk_size` from the fixed-chunking baseline. Keep K comparable to the
baseline chunk count for fair comparison.

### Why K is fixed (not input-adaptive) for stage 1

Input-adaptive K requires dynamic shapes through the RMT forward pass, which
complicates batching significantly. Fix K for now. This means "dynamic" refers to
boundary *positions* only, not chunk count. This is sufficient to test the core
hypothesis. Note this explicitly in the paper: depth is fixed, boundary positions
are learned.

### Forward pass

```
x: [B, L, D]  (token embeddings, pre-RMT)

1. boundary scoring
   h = boundary_encoder(x)          # [B, L, D_hidden]  lightweight, see below
   b = sigmoid(linear(h))           # [B, L, 1]  raw boundary score per position

2. soft chunk assignment
   # Compute cumulative boundary mass to assign tokens to chunk slots
   # Normalize b so that sum over L ≈ K (expected number of chunks = K)
   b_norm = b * (K / (b.sum(dim=1, keepdim=True) + eps))  # [B, L, 1]
   
   # Cumulative sum gives each token a "position in chunk space" [0, K]
   cumsum = b_norm.cumsum(dim=1)    # [B, L, 1]  ranges ~[0, K]
   
   # Soft assignment matrix: token i belongs to chunk j with weight
   # W[b,i,j] = exp(-0.5 * ((cumsum[b,i] - (j+0.5)) / sigma)^2)
   # j in [0, K-1], sigma is temperature hyperparameter
   j_centers = torch.arange(K, device=x.device).float() + 0.5  # [K]
   W = torch.exp(
       -0.5 * ((cumsum - j_centers.view(1, 1, K)) / sigma) ** 2
   )  # [B, L, K]
   W = W / (W.sum(dim=1, keepdim=True) + eps)  # normalize over tokens

3. chunk representations
   chunks = torch.einsum('blk,bld->bkd', W, x)  # [B, K, D]

4. return chunks, b.squeeze(-1)
```

### Boundary encoder: keep it lightweight

The boundary encoder must not dominate compute or add a separate pre-training step.
Two acceptable options in order of preference:

**Option A — causal 1D conv (preferred for stage 1):**
```python
boundary_encoder = nn.Sequential(
    nn.Conv1d(D, D_hidden, kernel_size=3, padding=1, groups=D_hidden//D),
    nn.GELU(),
    nn.Conv1d(D_hidden, D_hidden, kernel_size=1),
)
```
Causal convolution keeps the local context window small. Depthwise first layer is
parameter-efficient. D_hidden = D // 4 is a reasonable default.

**Option B — single linear (ablation baseline):**
```python
boundary_encoder = nn.Linear(D, D_hidden)
```
Use this in an ablation to show that the conv's local context helps boundary
prediction over pure per-token scoring.

Do not use a Transformer or Mamba as the boundary encoder at stage 1. This would
add significant compute and conflate the chunker's contribution with the encoder's
capacity. H-Net uses Mamba as encoder, but that is part of their full architecture.
Here the encoder is only for boundary scoring; the heavy lifting is done by RMT.

### Temperature sigma

`sigma` controls the sharpness of chunk assignments. Start with `sigma=1.0`.
At inference you can anneal toward hard boundaries (sigma → 0) or keep soft.
Soft boundaries at training, hard at inference is a reasonable default — implement
both and make it a flag.

Hard boundary inference:
```python
# argmax of boundary scores gives hard boundary positions
hard_boundaries = b.squeeze(-1).topk(K-1, dim=1).indices  # [B, K-1]
# use these to split x into K non-overlapping segments and mean-pool each
```

---

## Integration with RMT

### The single integration point

In the existing RMT forward pass, locate where the input sequence is chunked into
segments. It will look roughly like:

```python
# existing fixed chunking (find and identify this, do not delete)
chunks = x.unfold(1, chunk_size, chunk_size)  # or equivalent
# shape: [B, n_chunks, chunk_size, D]
# then each chunk is processed by the transformer with prepended memory tokens
```

Replace this with:

```python
# adaptive chunking
chunks, boundary_scores = self.chunker(x)  # [B, K, D], [B, L]
# chunks are already pooled to D-dim; each chunk is a single token for RMT
# memory tokens are prepended per-chunk as usual
```

The key difference: in fixed-chunking RMT, each "chunk" is a sequence of tokens
that the transformer attends over. In this adaptive version, each chunk is a
*single pooled token* of dimension D. The RMT processes a sequence of K pooled
chunk tokens, with memory tokens recurrently passed between them.

If the existing RMT processes each chunk with full self-attention over chunk_size
tokens, you need to decide: do you keep that, or collapse each chunk to one token?

**Stage 1 decision: collapse each chunk to one pooled token.**
This is the minimal version. The RMT backbone then operates over K tokens total
(the pooled chunks) plus memory tokens. This means you lose within-chunk
self-attention, which is a known tradeoff — document it. Within-chunk attention
can be added in stage 1.5 if results warrant.

Concretely after integration:
```
input:    [B, L, D]  (L original tokens)
chunker:  [B, K, D]  (K pooled chunks)
RMT:      K recurrent steps, each step processes 1 chunk token + memory tokens
output:   [B, K, D]  (RMT output per chunk)
```

For language modeling loss, the output needs to be projected back to token space.
See decoder section below.

### Memory token recurrence

This should not change at all. Memory tokens are initialized once, passed
left-to-right across K chunk steps, and the final memory state carries the
cross-chunk context. The only change is that "chunk steps" now correspond to
learned segment boundaries rather than fixed windows.

---

## Decoder (for language modeling)

If the task is sequence classification (ListOps, NLGraph), no decoder is needed —
use the final memory token state as the sequence representation and add a
classification head.

If you need token-level outputs (future stages), you will need a decoder that maps
chunk representations back to token space. For stage 1, skip this unless the task
requires it. Do not build the decoder until it is needed.

---

## New files to create

```
chunker/
  __init__.py
  dynamic_chunker.py      # DynamicChunker module
  hard_chunker.py         # hard boundary inference version
  utils.py                # boundary score visualization, alignment metrics

tasks/
  listops/
    dataset.py            # ListOps data loading and tokenization
    eval.py               # accuracy + boundary alignment metric
  nlgraph/
    dataset.py            # NLGraph connectivity/cycle loading
    eval.py               # accuracy + edge boundary alignment metric

experiments/
  train_listops.py        # training script, stage 1
  train_nlgraph.py
  ablations.py            # fixed vs adaptive chunker comparison
```

Modify only:
- `rmt/model.py` (or equivalent) — add `self.chunker = DynamicChunker(...)` and
  replace the fixed chunking line
- `rmt/config.py` (or equivalent) — add `use_adaptive_chunking: bool`,
  `n_chunks: int`, `chunker_hidden_dim: int`, `chunker_sigma: float`

---

## Tasks: stage 1 experiments

### Primary: ListOps

**Why this first:** ListOps has syntactic boundaries (brackets) that are
unambiguous ground truth. Fixed chunking fails *structurally* — a chunk boundary
inside a bracket pair makes the sub-expression computation impossible, not just
harder. This gives a clean controlled ablation.

**Input encoding:**
Tokenize at character level or use a small vocabulary that preserves bracket
tokens as single units: `(`, `)`, `MAX`, `MIN`, `SUM`, `MED`, `SM`, digits.
Do not use a subword tokenizer that might merge bracket tokens with neighbors.
Bracket tokens must be atomic in the vocabulary.

**Expected learned behavior:**
The chunker should learn to place high boundary scores at `(` tokens and/or `)` 
tokens. Verify this by plotting boundary scores vs. bracket positions on held-out
examples. If the learned boundaries do not correlate with brackets, the signal is
not reaching the chunker — check gradient flow through the soft assignment matrix.

**Boundary alignment metric:**
```python
def bracket_alignment(boundary_scores, input_ids, bracket_token_ids, topk):
    """
    For each sequence, take the topk boundary positions by score.
    Compute what fraction fall on bracket tokens.
    Baseline: random topk selection gives len(brackets)/L fraction.
    Report lift over baseline.
    """
```

**Baseline to beat:**
Fixed-chunk RMT with chunk_size tuned to best performance (try 4, 8, 16, 32).
The adaptive chunker should outperform the best fixed-chunk baseline, and the
improvement should grow with nesting depth.

**Dataset:** https://github.com/google-research/long-range-arena
Sequences up to length 2000. Use the standard train/val/test splits.

### Secondary: NLGraph connectivity

**Input encoding:**
One sentence per edge: "Node 0 is connected to node 1." Use a sentence tokenizer
that keeps each edge description roughly intact. The natural chunk is one sentence.

**Expected learned behavior:**
High boundary scores at sentence boundaries (`.` followed by `Node`).
Boundary alignment metric: fraction of top-K boundary positions that coincide
with sentence boundaries.

**Why after ListOps:** NLGraph has implicit (semantic) boundaries vs. ListOps's
explicit (syntactic) ones. Demonstrating generalization from syntactic to semantic
boundaries is a strong result. Run NLGraph only after ListOps results are stable.

**Subtask order within NLGraph:**
1. Connectivity (binary: is node A reachable from node B?) — simplest, binary label
2. Cycle detection (binary) — requires tracking full graph state
3. Shortest path (integer output) — skip for stage 1, needs decoder

### Diagnostic probe: Dyck-n

After training on ListOps, evaluate boundary scores on Dyck-n sequences (balanced
brackets at depth n) without any fine-tuning. If boundary scores align with bracket
structure on Dyck-n, the chunker has learned something structural, not task-specific.
This is a zero-shot transfer probe — cheap and compelling.

Generate Dyck-n sequences:
```python
def gen_dyck(n, length, depth_limit):
    # n bracket types, sequences of given length, max nesting depth_limit
    # returns (sequence, boundary_positions) where boundary_positions
    # are ground truth opening brackets
```

---

## Ablations to run (in order)

1. **Fixed vs. adaptive chunking** — same RMT, same K, same everything except
   fixed-size vs. dynamic boundaries. This is the primary result.

2. **Conv encoder vs. linear encoder** — isolates whether local context helps
   boundary prediction.

3. **Soft vs. hard boundaries at inference** — check that hard boundaries do not
   degrade performance vs. soft (if they do, sigma annealing schedule may be needed).

4. **K sensitivity** — vary K in {L//32, L//16, L//8, L//4}. If adaptive chunking
   is robust across K but fixed chunking degrades below optimal chunk_size, that
   is a strong argument for adaptive.

5. **Boundary score entropy** — measure entropy of boundary scores across training.
   Collapsing entropy (scores all near 0 or all near 1) indicates chunker collapse.
   Add this as a training monitor. If collapse happens, try adding an entropy
   regularizer: `loss += lambda_ent * (b * log(b + eps) + (1-b) * log(1-b + eps)).mean()`

---

## Training setup

No separate pre-training of the chunker. Train end-to-end from scratch on the
downstream task. The chunker's gradient comes entirely from task loss, backpropagated
through the soft assignment matrix W. Verify gradient flow explicitly:
```python
# after first backward pass, check
assert chunker.boundary_encoder.weight.grad is not None
assert chunker.boundary_encoder.weight.grad.abs().mean() > 0
```

If gradients to the chunker are vanishingly small, the soft assignment matrix is
too diffuse (sigma too large) or the task signal is not reaching it. Try:
- Reducing sigma
- Gradient clipping separately on chunker vs. RMT parameters
- Warmup: freeze chunker for first N steps, then unfreeze

Optimizer: AdamW with weight decay 0.01. Use a slightly higher learning rate for
the chunker than for RMT (2x is a reasonable starting point) since it is trained
from scratch while RMT may be initialized from a checkpoint.

---

## What NOT to do in stage 1

- Do not add a second chunking level (that is stage 2)
- Do not add byte-level input (that is stage 2)
- Do not add a decoder for token-level generation (not needed for classification tasks)
- Do not use Mamba or any SSM as the boundary encoder (adds too many variables)
- Do not add boundary supervision loss — the whole point is that boundaries are
  learned from task signal alone. If you add supervised boundary loss, you are
  testing a different hypothesis.
- Do not change the RMT memory mechanism, memory token count, or recurrence direction

---

## Key references

- RMT: Bulatov et al., "Recurrent Memory Transformer", NeurIPS 2022
- H-Net: Hwang, Wang, Gu, "Dynamic Chunking for End-to-End Hierarchical Sequence
  Modeling", arXiv 2507.07955 (2025) — read sections 3.1–3.3 for the soft
  assignment mechanism. The boundary scorer and cumsum-based pooling is the
  component being adapted here.
- Long Range Arena (ListOps): Tay et al., arXiv 2011.04006
- NLGraph: Wang et al., "Can Language Models Solve Graph Problems in Natural
  Language?", NeurIPS 2023, arXiv 2305.10037

---

## Definition of done for stage 1

- [ ] DynamicChunker implemented and tested on synthetic input
- [ ] RMT integration: single line change at chunking point, all existing tests pass
- [ ] ListOps training runs end-to-end
- [ ] Boundary alignment metric implemented and logged during training
- [ ] Ablation 1 (fixed vs. adaptive) run and results recorded
- [ ] Boundary score visualization on 10 held-out ListOps examples
- [ ] Chunker gradient flow verified (not vanishing)
- [ ] Dyck-n probe run (zero-shot boundary alignment)
- [ ] NLGraph connectivity experiment run