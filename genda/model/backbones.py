import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    ModernBertConfig, ModernBertForMaskedLM,
    LlamaConfig, LlamaModel,
)
from transformers.modeling_outputs import MaskedLMOutput
from typing import Optional, Literal
import math
import warnings


class GenDAModernBertConfig(ModernBertConfig):
    """Limit nested RoPE settings to active attention types, without global patches."""

    def standardize_rope_params(self):
        params = getattr(self, "rope_parameters", None)
        layer_types = getattr(self, "layer_types", None)
        if isinstance(params, dict) and layer_types and all(isinstance(v, dict) for v in params.values()):
            self.rope_parameters = {k: v for k, v in params.items() if k in layer_types}
        return super().standardize_rope_params()


class GenDA_ModernBERT(nn.Module):
    def __init__(self, tokenizer, hidden_dim, n_layers, dropout_rate,
                 num_attention_heads, max_position_embeddings, attn_implementation,
                 global_attn_every_n_layers, local_attention,
                 position_encoding: Literal["rope", "none", "learned"] = "rope"):
        """
        Parameters
        ----------
        position_encoding : str
            "rope"    — Rotary Position Embeddings (default, ModernBERT native).
            "none"    — No positional encoding.  RoPE replaced with identity.
            "learned" — Learned absolute position embeddings that replace RoPE.
        """
        super().__init__()

        self.position_encoding = position_encoding

        # 2. Configuration for ModernBERT
        # We explicitly set "flash_attention_2" to fix memory issues at 8k length.
        config = GenDAModernBertConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=hidden_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=hidden_dim * 4,
            
            # --- The Fix for Small Vocabs ---
            pad_token_id=tokenizer.pad_id,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            cls_token_id=tokenizer.token_to_id.get("<CLS>", tokenizer.bos_id),
            sep_token_id=tokenizer.token_to_id.get("<SEP>", tokenizer.eos_id),           
            # --- Regularization & Dropouts ---
            hidden_dropout_prob=dropout_rate, # Randomly drops neurons inside the 16 transformer layers.
            attention_probs_dropout_prob=0.0, # Keep this 0.0 for Flash Attention efficiency!
            classifier_dropout=dropout_rate,  # Randomly drops neurons in the final decision layer.

            # ModernBERT uses RoPE natively.
            # NOTE: rope_parameters is set via object.__setattr__ after construction
            # to avoid @strict validation crashes in some transformers versions.
            
            # --- The Speed Secret ---
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16, # Fixes Flash Attention 2 dtype warning
            
            # enforce globan attention cuz it seems that global attentinon matters in DLM
            global_attn_every_n_layers=global_attn_every_n_layers,
            local_attention=local_attention,

            # Explicitly suppress rope_parameters → must be None to pass validation
            rope_parameters=None,

        )
        
        # Now inject the correct rope_parameters *after* the config is created,
        # bypassing the strict dataclass setter so we don't trigger the buggy standardisation.
        rope_params = {
            "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        }
        object.__setattr__(config, "rope_parameters", rope_params)

        # Clean up the deprecated top‑level attributes so nothing tries to use them
        object.__setattr__(config, "rope_type", None)
        object.__setattr__(config, "rope_theta", None)

        # 2. The Backbone + Head Combined
        # ModernBertForMaskedLM includes the .bert encoder AND the .cls prediction head.
        self.model = ModernBertForMaskedLM(config)
        self.config = self.model.config

        # ── Learned absolute position embeddings ──
        if position_encoding == "learned":
            self.abs_pos_embed = nn.Embedding(max_position_embeddings, hidden_dim)
        else:
            self.abs_pos_embed = None

        # ── Disable RoPE for non-RoPE modes ──
        if position_encoding in ("learned", "none"):
            self._disable_modernbert_rope()

    # ═══════════════════════════════════════════════════════════════════════
    def _disable_modernbert_rope(self):
        """Replace ModernBERT's rotary embedding with identity (cos=1, sin=0)."""
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        self.model.model.rotary_emb = _ModernBertNoOpRotaryEmbedding(head_dim)

    def enable_gradient_checkpointing(self):
        self.model.gradient_checkpointing_enable() # SAVE MEM, BUT SLOWER   
        self.model.config.use_cache = False

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, output_attentions=False, **kwargs):
        if self.position_encoding == "learned":
            bsz, seq_len = input_ids.shape
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            pos_embeds = self.abs_pos_embed(positions)
            inputs_embeds = self.model.model.embeddings(input_ids=input_ids) + pos_embeds
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        return outputs



# ═══════════════════════════════════════════════════════════════════════════
#  Positional Encoding Utilities
# ═══════════════════════════════════════════════════════════════════════════

class _NoOpRotaryEmbedding(nn.Module):
    """Returns cos=1, sin=0 → identity rotation (no positional encoding).

    The HF LlamaRotaryEmbedding returns (cos, sin) with shape
    [batch_size, seq_len, head_dim].  We replicate this shape to ensure
    broadcasting inside apply_rotary_pos_emb works identically.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids):
        bsz, seq_len = x.shape[0], x.shape[1]
        cos = torch.ones((bsz, seq_len, self.head_dim), device=x.device, dtype=x.dtype)
        sin = torch.zeros((bsz, seq_len, self.head_dim), device=x.device, dtype=x.dtype)
        return cos, sin


class _ModernBertNoOpRotaryEmbedding(nn.Module):
    """Replaces ModernBertRotaryEmbedding with identity (cos=1, sin=0).

    Used when position_encoding is 'learned' or 'none' so that RoPE
    does not inject additional positional signal.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids, layer_type=None):
        bsz, seq_len = x.shape[0], x.shape[1]
        cos = torch.ones((bsz, seq_len, self.head_dim), device=x.device, dtype=x.dtype)
        sin = torch.zeros((bsz, seq_len, self.head_dim), device=x.device, dtype=x.dtype)
        return cos, sin


def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    ALiBi slopes: 2^(-8*i/n) for i in [1..n], as per the original paper.
    These slopes are head-specific and multiply the negative absolute distance.

    Example: for 8 heads → [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    """
    # Geometric sequence: start at 2^(-8/n), ratio = 2^(-8/n)
    ratio = 2 ** (-8.0 / num_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, num_heads + 1)])
    return slopes


def _make_alibi_forward(num_heads: int, num_kv_heads: int, head_dim: int,
                         dropout_rate: float, slopes: torch.Tensor):
    """
    Build a custom forward() for a LlamaAttention module that uses ALiBi
    instead of RoPE.  This forward skips rotary embeddings entirely and
    computes the ALiBi bias, then calls SDPA directly.

    NOTE: ALiBi is NOT compatible with flash_attention_2.  We force eager/SDPA
    by calling F.scaled_dot_product_attention ourselves.
    """
    def alibi_forward(self, hidden_states, attention_mask=None, position_ids=None,
                       past_key_value=None, output_attentions=False, use_cache=False,
                       cache_position=None, position_embeddings=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        # --- ALiBi bias ---
        device = hidden_states.device
        positions = torch.arange(q_len, device=device, dtype=torch.float32)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)          # [L, L]
        alibi_bias = -slopes.to(device).unsqueeze(1).unsqueeze(1) * rel_pos.abs().unsqueeze(0)  # [n_heads, 1, L, L]

        # --- Combine ALiBi with external padding/causal mask ---
        # attention_mask is a 4D float [B, 1, L, L]:  0.0 for valid, -inf for masked.
        # alibi_bias is [n_heads, 1, L, L].  Adding them broadcasts to [B, n_heads, L, L].
        final_mask = alibi_bias
        if attention_mask is not None:
            final_mask = final_mask + attention_mask

        # --- GQA: expand KV heads to match Q heads ---
        if num_kv_heads < num_heads:
            ratio = num_heads // num_kv_heads
            key_states   = key_states.repeat_interleave(ratio, dim=1)
            value_states = value_states.repeat_interleave(ratio, dim=1)

        # --- SDPA with combined additive mask ---
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=final_mask,               # float → additive
            dropout_p=dropout_rate if self.training else 0.0,
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None   # (output, attn_weights) — 2-tuple expected by LlamaDecoderLayer

    return alibi_forward


class GenDA_Llama(nn.Module):
    def __init__(self, tokenizer, hidden_dim, n_layers, dropout_rate,
                 num_attention_heads, max_position_embeddings, attn_implementation,
                 intermediate_size=None, num_kv_heads=None, rope_theta=10000.0,
                 position_encoding: Literal["rope", "alibi", "none", "learned"] = "rope"):
        """
        Parameters
        ----------
        position_encoding : str
            "rope"    — Rotary Position Embeddings (default, current behaviour).
            "alibi"   — ALiBi (Attention with Linear Biases).  Translation-invariant.
                         Forces SDPA attention (not Flash-Attention-2 compatible).
            "none"    — No positional encoding.  RoPE replaced with identity.
            "learned" — Learned absolute position embeddings that replace RoPE.
        """
        super().__init__()

        self.position_encoding = position_encoding
        self.head_dim = hidden_dim // num_attention_heads
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_kv_heads if num_kv_heads is not None else num_attention_heads

        # ── ALiBi pre-checks ──
        if position_encoding == "alibi" and attn_implementation == "flash_attention_2":
            warnings.warn(
                "ALiBi is not compatible with flash_attention_2. "
                "Forcing attn_implementation='sdpa'.  Training will be slower.",
                stacklevel=2,
            )
            attn_implementation = "sdpa"

        config = LlamaConfig(
            vocab_size=len(tokenizer.vocab),
            hidden_size=hidden_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=self.n_kv_heads,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size if intermediate_size is not None else hidden_dim * 4,
            rope_theta=rope_theta,
            pad_token_id=tokenizer.pad_id,
            bos_token_id=tokenizer.bos_id,
            eos_token_id=tokenizer.eos_id,
            attention_dropout=dropout_rate,
            torch_dtype=torch.bfloat16,
        )
        config._attn_implementation = attn_implementation

        # ── Learned absolute position embeddings (before backbone so hooks can find it) ──
        if position_encoding == "learned":
            self.abs_pos_embed = nn.Embedding(max_position_embeddings, hidden_dim)
        else:
            self.abs_pos_embed = None

        # Use standard LlamaModel — no monkey-patching needed.
        self.backbone = LlamaModel(config)
        self.config = self.backbone.config

        # CRITICAL: Set is_causal=False on every attention layer.
        # Both SDPA (sdpa_attention.py:85) and Flash Attention (flash_attention.py:55)
        # check module.is_causal explicitly, so this properly disables causal masking.
        for layer in self.backbone.layers:
            layer.self_attn.is_causal = False

        # ── Apply position-encoding modifications ──
        self._apply_position_encoding(position_encoding, dropout_rate)

        self.head = nn.Linear(hidden_dim, len(tokenizer.vocab), bias=False)

    # ═══════════════════════════════════════════════════════════════════════
    def _apply_position_encoding(self, pe_type: str, dropout_rate: float):
        """Post-init modifications to replace / augment the positional encoding."""
        if pe_type == "none":
            self._disable_rope()
        elif pe_type == "alibi":
            self._install_alibi(dropout_rate)
        elif pe_type == "learned":
            self._disable_rope()
            # No pre-forward hook — position embeddings are added directly in forward().

    # ═══════════════════════════════════════════════════════════════════════
    def _disable_rope(self):
        """Replace RoPE with identity → no positional encoding."""
        for layer in self.backbone.layers:
            layer.self_attn.rotary_emb = _NoOpRotaryEmbedding(self.head_dim)

    # ═══════════════════════════════════════════════════════════════════════
    def _install_alibi(self, dropout_rate: float):
        """Replace each attention module's forward with one that uses ALiBi."""
        slopes = _get_alibi_slopes(self.n_heads)
        for layer in self.backbone.layers:
            layer.self_attn._alibi_slopes = slopes
            new_forward = _make_alibi_forward(
                self.n_heads, self.n_kv_heads, self.head_dim,
                dropout_rate, slopes,
            )
            # Bind as bound method so 'self' works inside
            import types
            layer.self_attn.forward = types.MethodType(new_forward, layer.self_attn)

    def enable_gradient_checkpointing(self):
        self.backbone.gradient_checkpointing_enable()
        self.backbone.config.use_cache = False

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, output_attentions=False, **kwargs):
        bsz, seq_len = input_ids.shape

        # CRITICAL: HF's LlamaModel internally calls create_causal_mask() which
        # always produces a lower-triangular causal mask.  We bypass this by
        # passing a pre-computed 4D mask (which triggers an early-exit in
        # _preprocess_mask_arguments).  HOWEVER, Flash Attention 2 calls
        # .nonzero() on the mask for its "unpadding" optimisation, and a 4D
        # mask of shape [128, 1, 4096, 4096] has 2.1B elements → overflows
        # INT_MAX.  When there is no padding we pass None instead — the
        # is_causal=False on every attention module keeps SDPA bidirectional.
        has_padding = attention_mask is not None and not attention_mask.all()
        if attention_mask is None or not has_padding:
            # No padding: skip the 4D mask.  is_causal=False on every self_attn
            # module keeps Flash Attention 2 / SDPA bidirectional.
            final_mask = None
        else:
            # Mixed-length batch: build 4D [B, 1, L, L] additive mask.
            # 0.0 = attend, -inf = mask.
            valid = attention_mask.to(torch.float32)       # [B, L]
            final_mask = valid[:, None, None, :] * valid[:, None, :, None]  # [B, 1, L, L], 0 or 1
            final_mask = (1.0 - final_mask) * torch.finfo(torch.float32).min   # 0 → 0.0, 1 → -inf

        # ── Learned PE: inject position embeddings before the backbone ──
        if self.position_encoding == "learned":
            positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
            pos_embeds = self.abs_pos_embed(positions)
            hidden = self.backbone.embed_tokens(input_ids) + pos_embeds
            backbone_out = self.backbone(
                inputs_embeds=hidden, attention_mask=final_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        else:
            backbone_out = self.backbone(
                input_ids=input_ids, attention_mask=final_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        hidden = backbone_out.last_hidden_state
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_out.hidden_states if output_hidden_states else None,
            attentions=backbone_out.attentions if output_attentions else None,
        )

