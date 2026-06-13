import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

class NeoRexConfig(PretrainedConfig):
    model_type = "neorex"
    def __init__(
        self,
        vocab_size: int = 32778, # Mistral vocab + 10 agentic tokens
        d_model: int = 1280,
        n_layers: int = 24,
        n_heads: int = 20,
        latent_dim: int = 256,
        mlp_hidden: int = 3456,
        max_position_embeddings: int = 131072, # 128k
        sliding_window: int = 1024,
        rope_theta: float = 10000.0,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        mtp_depth: int = 3, # Multi-token prediction depth
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.mlp_hidden = mlp_hidden
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.mtp_depth = mtp_depth
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (self.weight.shape[0],), self.weight, self.eps)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=131072, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            # Dynamically extend if needed
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # q, k shape: [bs, num_heads, seq_len, head_dim]
    # position_ids: [bs, seq_len]
    cos = cos.squeeze(0).squeeze(0) # [seq_len, head_dim]
    sin = sin.squeeze(0).squeeze(0)
    cos = cos[position_ids].unsqueeze(1) # [bs, 1, seq_len, head_dim]
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    return q_embed, None

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, in_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SlidingWindowAttention(nn.Module):
    def __init__(self, config: NeoRexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.sliding_window = config.sliding_window
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_kv = (k, v) if use_cache else None

        # SWA Masking: Only attend to the last `sliding_window` tokens
        swa_mask = torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=hidden_states.device)
        swa_mask = torch.tril(swa_mask, diagonal=kv_seq_len - q_len)
        window_mask = torch.triu(torch.ones_like(swa_mask), diagonal=kv_seq_len - q_len - self.sliding_window + 1)
        swa_mask = swa_mask & window_mask

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = swa_mask & attention_mask
        else:
            if q_len == 1 and kv_seq_len <= self.sliding_window:
                attention_mask = None
            else:
                attention_mask = swa_mask

        # SDPA handles custom boolean masks efficiently
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask, 
            is_causal=False # SWA mask is explicitly passed
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        return attn_output, new_kv

class MultiHeadLatentAttention(nn.Module):
    """
    MLA compresses K and V into a single latent vector to drastically reduce KV cache size.
    """
    def __init__(self, config: NeoRexConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = self.d_model // self.n_heads
        self.latent_dim = config.latent_dim
        
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.kv_a_proj = nn.Linear(self.d_model, self.latent_dim, bias=False)
        self.kv_a_norm = RMSNorm(self.latent_dim)
        
        # Up-project from latent to multi-head K and V
        self.k_up_proj = nn.Linear(self.latent_dim, self.d_model, bias=False)
        self.v_up_proj = nn.Linear(self.latent_dim, self.d_model, bias=False)
        self.o_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compress KV to latent
        compressed_kv = self.kv_a_proj(hidden_states)
        compressed_kv = self.kv_a_norm(compressed_kv)
        
        # In inference with cache, we ONLY cache `compressed_kv`
        if past_key_value is not None:
            compressed_kv = torch.cat([past_key_value[0], compressed_kv], dim=1)
            
        new_kv = (compressed_kv,) if use_cache else None

        # Decompress for attention
        k = self.k_up_proj(compressed_kv).view(bsz, compressed_kv.size(1), self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_up_proj(compressed_kv).view(bsz, compressed_kv.size(1), self.n_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = k.shape[-2]
        cos, sin = self.rotary_emb(v, seq_len=kv_seq_len)
        
        # We only apply RoPE to Q and K after decompression.
        # Since K was decompressed from the *entire* sequence latent cache, we need to pass the full position_ids 
        # corresponding to the cache to properly rotate K.
        if past_key_value is not None:
            # During generation (q_len=1), K contains past+current. 
            # We must apply RoPE to all of K using full positions, taking into account any position offsets.
            offsets = position_ids[:, -1:] - kv_seq_len + 1
            steps = torch.arange(kv_seq_len, device=hidden_states.device).unsqueeze(0)
            full_position_ids = offsets + steps
            _, k = apply_rotary_pos_emb(q, k, cos, sin, full_position_ids)
            q, _ = apply_rotary_pos_emb(q, None, cos, sin, position_ids)
        else:
            q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

        # Causal masking for global attention
        causal_mask = torch.ones((q_len, kv_seq_len), dtype=torch.bool, device=hidden_states.device)
        causal_mask = torch.tril(causal_mask, diagonal=kv_seq_len - q_len)

        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                attention_mask = attention_mask.bool()
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            attention_mask = causal_mask & attention_mask
            is_causal = False
        else:
            if q_len > 1:
                attention_mask = None
                is_causal = True
            else:
                attention_mask = None
                is_causal = False

        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask, 
            is_causal=is_causal
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        return attn_output, new_kv

class NeoRexBlock(nn.Module):
    def __init__(self, config: NeoRexConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        # Interleave: 3 SWA, 1 MLA
        self.is_global = (layer_idx % 4 == 3)
        
        if self.is_global:
            self.attention = MultiHeadLatentAttention(config)
        else:
            self.attention = SlidingWindowAttention(config)
            
        self.mlp = SwiGLU(config.d_model, config.mlp_hidden)
        self.input_layernorm = RMSNorm(config.d_model)
        self.post_attention_layernorm = RMSNorm(config.d_model)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, use_cache=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        attn_output, new_kv = self.attention(
            hidden_states, 
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            past_key_value=past_key_value, 
            use_cache=use_cache
        )
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, new_kv

class NeoRexModel(PreTrainedModel):
    config_class = NeoRexConfig
    def __init__(self, config: NeoRexConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([NeoRexBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False):
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
            if past_key_values is not None:
                val = past_key_values[0][0]
                past_len = val.shape[-2] if val.dim() == 4 else val.shape[1]
                position_ids = position_ids + past_len
                
        hidden_states = self.embed_tokens(input_ids)
        next_cache = () if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_kv = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_kv,
                use_cache=use_cache
            )
            if use_cache:
                next_cache += (new_kv,)
                
        hidden_states = self.norm(hidden_states)
        return hidden_states, next_cache

class NeoRexForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NeoRexConfig
    def __init__(self, config: NeoRexConfig):
        super().__init__(config)
        self.model = NeoRexModel(config)
        
        # Primary LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # MTP (Multi-Token Prediction) Heads
        self.mtp_depth = config.mtp_depth
        if self.mtp_depth > 0:
            self.mtp_heads = nn.ModuleList([
                nn.Sequential(
                    RMSNorm(config.d_model),
                    nn.Linear(config.d_model, config.vocab_size, bias=False)
                ) for _ in range(self.mtp_depth)
            ])
            
        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            if self.mtp_depth > 0:
                for mtp_head in self.mtp_heads:
                    mtp_head[1].weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = value.weight
            if self.config.mtp_depth > 0:
                for mtp_head in self.mtp_heads:
                    mtp_head[1].weight = value.weight

    def forward(self, input_ids, labels=None, attention_mask=None, position_ids=None, past_key_values=None, use_cache=False, return_dict=True):
        hidden_states, next_cache = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Standard Next Token Prediction Loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            # MTP Loss
            if self.mtp_depth > 0:
                for i, mtp_head in enumerate(self.mtp_heads):
                    # mtp_depth=1 -> predicts token at t+2
                    shift_amount = i + 2
                    if logits.size(1) > shift_amount:
                        mtp_logits = mtp_head(hidden_states)[..., :-shift_amount, :].contiguous()
                        mtp_labels = labels[..., shift_amount:].contiguous()
                        mtp_loss = loss_fct(mtp_logits.view(-1, self.config.vocab_size), mtp_labels.view(-1))
                        # Weight the MTP loss (e.g. equally or discounted)
                        loss += mtp_loss * 0.5 
            
            # Scale the total loss down so it aligns with standard single-token loss magnitudes
            # This avoids artificially inflating the displayed loss due to the auxiliary heads!
            if self.mtp_depth > 0:
                loss = loss / (1.0 + self.mtp_depth * 0.5)
            
        if not return_dict:
            return (loss, logits, next_cache) if loss is not None else (logits, next_cache)
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
        )
