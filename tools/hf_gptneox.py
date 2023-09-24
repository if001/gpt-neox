from transformers.models.gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel, GPTNeoXLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXMLP, GPTNeoXAttention
from transformers.activations import ClassInstantier, ACT2CLS
from torch import Tensor, nn
import torch

from typing import Callable, Optional, Tuple
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
    
class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x) * x

# ACT2CLS['swiglu'] = SwiGLUFFN
ACT2CLS['swiglu'] = SwiGLU
ACT2FN = ClassInstantier(ACT2CLS)

class GPTNeoX2MLP(GPTNeoXMLP):
    def __init__(self, config):
        _copy_hidden_act = config.hidden_act
        config.hidden_act = "gelu"
        super().__init__(config)

        config.hidden_act = _copy_hidden_act
        self.act = ACT2FN[config.hidden_act]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, cos_k=None, sin_k=None):
    """
    q, k: [bs, num_heads, seq_len, rot_dim]
    cos, sin: [seq_len, rot_dim / 2]
    position_ids: [bs, seq_len]
    """
    # print(f"q: {q.shape}, k: {k.shape}, cos: {cos.shape}, sin: {sin.shape}, position_ids: {position_ids.shape}")
    import einops
    cos = einops.repeat(cos, 's r -> s (2 r)')
    sin = einops.repeat(sin, 's r -> s (2 r)')
    cos_k = einops.repeat(cos_k, 's r -> s (2 r)')
    sin_k = einops.repeat(sin_k, 's r -> s (2 r)')
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, rot_dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, rot_dim]
    cos_k = cos_k[position_ids].unsqueeze(1)  # [bs, 1, seq_len, rot_dim]
    sin_k = sin_k[position_ids].unsqueeze(1)  # [bs, 1, seq_len, rot_dim]

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed

class RotaryEmbedding(torch.nn.Module):
    """Based on Tri Dao's XPos: https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/layers/rotary.py"""
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: int = 10_000,
        scale_base: int = 512,
        device: str = None
    ):
        super().__init__()
        self.dim = dim
        self.seq_len_cached = max_position_embeddings

        # Set up `inv_freq` term
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Set up `scale` term
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None else None
        )
        self.register_buffer("scale", scale)

        # Seet up `cos..` and `sin...` cache terms
        t = torch.arange(self.seq_len_cached, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        # freqs = torch.cat((freqs, freqs), dim=-1)
        seq_range = torch.arange(self.seq_len_cached, dtype=self.scale.dtype, device=self.scale.device)
        power = (seq_range - self.seq_len_cached // 2) / self.scale_base
        scale_cached = self.scale.to(device=power.device) ** power.unsqueeze(-1)
        # scale_cached = torch.cat((scale_cached, scale_cached), dim=-1)
        self.register_buffer("cos_cached", torch.cos(freqs) * scale_cached, persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs) * scale_cached, persistent=False)
        self.register_buffer("cos_k_cached", torch.cos(freqs) / scale_cached, persistent=False)
        self.register_buffer("sin_k_cached", torch.sin(freqs) / scale_cached, persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1)
            seq_range = torch.arange(self.seq_len_cached, dtype=self.scale.dtype, device=self.scale.device)
            power = (seq_range - self.seq_len_cached // 2) / self.scale_base
            scale_cached = self.scale.to(device=power.device) ** power.unsqueeze(-1)
            scale_cached = torch.cat((scale_cached, scale_cached), dim=-1)
            self.register_buffer("cos_cached", torch.cos(freqs) * scale_cached, persistent=False)
            self.register_buffer("sin_cached", torch.sin(freqs) * scale_cached, persistent=False)
            self.register_buffer("cos_k_cached", torch.cos(freqs) / scale_cached, persistent=False)
            self.register_buffer("sin_k_cached", torch.sin(freqs) / scale_cached, persistent=False)
        return (
            self.cos_cached[:seq_len, ...],
            self.sin_cached[:seq_len, ...],
            self.cos_k_cached[:seq_len, ...],
            self.sin_k_cached[:seq_len, ...],
        )

class GPTNeoX2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them"
            )
        self.head_size = self.hidden_size // self.num_attention_heads

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.rotary_ndims = int(self.head_size * config.rotary_pct)
        print('config', config)
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rotary_emb_base,
            scale_base=config.rotary_scale_base,
        )

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_size, dtype=torch.float32)).to(torch.get_default_dtype()),
            persistent=False,
        )

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        has_layer_past = layer_past is not None

        # Compute QKV
        # Attention heads [batch, seq_len, hidden_size]
        #   --> [batch, seq_len, (np * 3 * head_size)]
        qkv = self.query_key_value(hidden_states)

        # [batch, seq_len, (num_heads * 3 * head_size)]
        #   --> [batch, seq_len, num_heads, 3 * head_size]
        new_qkv_shape = qkv.size()[:-1] + (self.num_attention_heads, 3 * self.head_size)
        qkv = qkv.view(*new_qkv_shape)

        # [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        query = qkv[..., : self.head_size].permute(0, 2, 1, 3)
        key = qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        value = qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        # Compute rotary embeddings on rotary_ndims
        query_rot = query[..., : self.rotary_ndims]
        query_pass = query[..., self.rotary_ndims :]
        key_rot = key[..., : self.rotary_ndims]
        key_pass = key[..., self.rotary_ndims :]

        # Compute token offset for rotary embeddings (when decoding)
        kv_seq_len = key.shape[-2]
        if has_layer_past:
            kv_seq_len += layer_past[0].shape[-2]

        # Add rotary embeddings to query and key
        # TODO: Check if using xpos
        cos, sin, cos_k, sin_k = self.rotary_emb(value, seq_len=kv_seq_len)
        query, key = apply_rotary_pos_emb(
            query_rot, key_rot, cos, sin, position_ids, cos_k=cos_k, sin_k=sin_k)

        query = torch.cat((query, query_pass), dim=-1)
        key = torch.cat((key, key_pass), dim=-1)

        # Cache QKV values
        if has_layer_past:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        present = (key, value) if use_cache else None

        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # Merge attn_head_size dim and num_attn_heads dim into hidden dim
        # [bs, seq_len, num_attention_heads, attn_head_size]
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(attn_output.size(0), attn_output.size(1), self.num_attention_heads * self.head_size)

        attn_output = self.dense(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        # q, k, v: [bs, num_attention_heads, seq_len, attn_head_size]
        # compute causal mask from causal mask buffer

        batch_size, num_attention_heads, query_length, attn_head_size = query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        query = query.view(batch_size * num_attention_heads, query_length, attn_head_size)
        key = key.view(batch_size * num_attention_heads, key_length, attn_head_size)
        attn_scores = torch.zeros(
            batch_size * num_attention_heads,
            query_length,
            key_length,
            dtype=query.dtype,
            device=key.device,
        )
        attn_scores = torch.baddbmm(
            attn_scores,
            query,
            key.transpose(1, 2),
            beta=1.0,
            alpha=(torch.tensor(1.0, dtype=self.norm_factor.dtype, device=self.norm_factor.device) / self.norm_factor),
        )
        attn_scores = attn_scores.view(batch_size, num_attention_heads, query_length, key_length)

        mask_value = torch.finfo(attn_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores = torch.where(causal_mask, attn_scores, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_scores = attn_scores + attention_mask

        # NOTE: Upcast to float32
        attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(value)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights


def attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(~ltor_mask, torch.finfo(attention_scores.dtype).min)
    return attention_scores


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param dim: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = dim
        self.p = p
        self.bias = bias

        self.scale = torch.nn.Parameter(torch.ones(dim))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = torch.nn.Parameter(torch.zeros(dim))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class GPTNeoX2Layer(GPTNeoXLayer):
    def __init__(self, config):
        _copy_hidden_act = config.hidden_act
        config.hidden_act = "gelu"
        super().__init__(config)

        config.hidden_act = _copy_hidden_act
        self.use_parallel_residual = config.use_parallel_residual
        # self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)

        # self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.attention = GPTNeoXAttention(config)
        self.attention = GPTNeoX2Attention(config)
        self.mlp = GPTNeoX2MLP(config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ):
        attention_layer_outputs = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attention_layer_outputs[0]  # output_attn: attn_output, present, (attn_weights)
        outputs = attention_layer_outputs[1:]

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            hidden_states = mlp_output + attn_output

        if use_cache:
            outputs = (hidden_states,) + outputs  # hidden_states, present, (attn_weights)
        else:
            outputs = (hidden_states,) + outputs[1:]  # hidden_states, (attn_weights)

        return outputs

class GPTNeoX2Model(GPTNeoXModel):
    def __init__(self, config):
        _copy_hidden_act = config.hidden_act
        config.hidden_act = "gelu"
        super().__init__(config)

        config.hidden_act = _copy_hidden_act
        self.layers = nn.ModuleList([GPTNeoX2Layer(config) for _ in range(config.num_hidden_layers)])

class GPTNeoX2ForCausalLM(GPTNeoXPreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.gpt_neox = GPTNeoX2Model(config)