import warnings
from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn
import torch.distributed as dist

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.utils import add_prefix
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
from sglang.srt.layers.layernorm import RMSNorm


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim) # BSND->BNSD
    sin = sin[position_ids].unsqueeze(unsqueeze_dim) # BSND->BNSD

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DeepseekV3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: PretrainedConfig,
            hidden_size: int,
            num_heads: int,
            qk_nope_head_dim: int,
            qk_rope_head_dim: int,
            v_head_dim: int,
            q_lora_rank: int,
            kv_lora_rank: int,
            rope_theta: float = 10000,
            rope_scaling: Optional[Dict[str, Any]] = None,
            max_position_embeddings: int = 8192,
            quant_config: Optional[Any] = None,
            reduce_results: bool = True,
            layer_id: int = None,
            prefix: str = "",
            alt_stream: Optional[torch.npu.Stream] = None,
    ) -> None:
        super().__init__()
        # self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.world_size = attn_tp_size

        self.config = config
        self.layer_idx = layer_id

        self.attention_dropout = config.attention_dropout
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_heads_per_rank = self.num_heads // self.world_size
        self.num_key_value_heads_per_rank = self.num_heads_per_rank

        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads_per_rank * self.q_head_dim, bias=False
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size, self.q_lora_rank, bias=config.attention_bias
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads_per_rank * self.q_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
            )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        # self.kv_b_proj_w_k = nn.Parameter(
        #     torch.zeros(self.num_heads_per_rank, self.qk_nope_head_dim, self.kv_lora_rank)
        # )
        # self.kv_b_proj_w_v = nn.Parameter(
        #     torch.zeros(self.num_heads_per_rank, self.kv_lora_rank, self.v_head_dim)
        # )

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            )

        self.o_proj = nn.Linear(
            self.num_heads_per_rank * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            )

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False


        self.softmax_scale = self.q_head_dim ** (-0.5)
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None:
            mscale_all_dim = self.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def bmm_5d(self, x, y):
        b, s, n, _, d = x.shape
        x = x.view(b * s, n, d).transpose(0, 1) # n, bs, d
        output = torch.matmul(x, y) # n, bs, rank
        output = output.transpose(1, 0).view(b, s, n, -1)
        return output

    def prepare_qkv(
            self,
            hidden_states: torch.Tensor,
            cos_sin: torch.Tensor = None,
            kv_len: torch.IntTensor = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        q = q.view(bsz, q_len, self.num_heads_per_rank, self.q_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = q_pe.transpose(1, 2)
        q_nope = self.bmm_5d(
            q_nope.view(bsz, q_len, self.num_heads_per_rank, 1, self.qk_nope_head_dim),
            self.kv_b_proj_w_k
        )
        q_nope = q_nope.view(bsz, q_len, self.num_heads_per_rank, self.kv_lora_rank)
        q_nope = q_nope.transpose(1, 2)

        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_nope = (
            self.kv_a_layernorm(compressed_kv)
            .view(bsz, -1, 1, self.kv_lora_rank)
            .transpose(1, 2)
        ) # (bs, 1, q_len, kv_lora_rank)

        # rope
        cos, sin = cos_sin
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = torch.cat([q_nope, q_pe], dim=-1)
        key_states = torch.cat([k_nope, k_pe], dim=-1)

        kv_seq_len = k_nope.shape[-2]
        if past_key_value is not None:
            past_key_states = past_key_value[self.layer_idx][0]
            torch_npu.scatter_update_(past_key_states, kv_len, key_states, -2)
            if q_len == 1:
                key_states = past_key_states
            kv_seq_len = past_key_value[0][0].size()[-2]
        value_states = key_states
        return query_states, key_states, value_states, kv_seq_len

    def apply_attention_npu(
            self,
            query_states, key_states, value_states, kv_seq_len,
            attention_mask: Optional[torch.Tensor] = None,
            actual_seq_lengths_kv: list = None,
            output_attentions: bool = False,
            past_key_value: Optional[Cache] = None,
    ):
        # repeat k/v heads if n_kv_heads < n_heads
        bsz, _, q_len, _ = query_states.size()
        attn_weights = (
                torch.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale
        )
        assert attention_mask is not None
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        value_states = value_states[..., :self.kv_lora_rank]
        attn_output = torch.matmul(attn_weights, value_states)

        # kv rank opt
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = self.bmm_5d(
            attn_output.unsqueeze(3),
            self.kv_b_proj_w_v
        ) # (bs, q_len, num_heads, kv_lora_rank)
        attn_output = self.o_proj(attn_output.reshape(bsz, q_len, -1))
        if self.world_size > 1:
            dist.all_reduce(attn_output)
        return attn_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            kv_len: torch.IntTensor = None,
            actual_seq_lengths_kv: list = None,
            cos_sin: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        query_states, key_states, value_states, kv_seq_len = self.prepare_qkv(
            hidden_states=hidden_states,
            cos_sin=cos_sin,
            kv_len=kv_len,
            position_ids=position_ids,
            past_key_value=past_key_value
        )
        output = self.apply_attention_npu(
            query_states=query_states, key_states=key_states, value_states=value_states,
            kv_seq_len=kv_seq_len,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            past_key_value=past_key_value
        )
        return output
