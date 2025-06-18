# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from sglang.srt.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from sglang.srt.managers import expert_location_dispatch
from sglang.srt.managers.expert_distribution import (
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
)
from sglang.srt.managers.expert_location_dispatch import (
    ExpertLocationDispatchInfo,
    topk_ids_logical_to_physical,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.utils import get_compiler_backend, is_cuda, is_hip, is_npu, log_info_on_rank0, logger

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if _is_cuda:
    from sgl_kernel import moe_fused_gate

if _is_cuda or _is_hip:
    from sgl_kernel import topk_softmax

if _is_npu:
    import torch_npu

def fused_topk_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    M, _ = hidden_states.shape

    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    token_expert_indicies = torch.empty(
        M, topk, dtype=torch.int32, device=hidden_states.device
    )

    topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output.float(),
    )
    del token_expert_indicies

    return _fused_topk_postprocess(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        renormalize=renormalize,
        expert_location_dispatch_info=expert_location_dispatch_info,
        num_token_non_padded=num_token_non_padded,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _fused_topk_postprocess(
    topk_weights,
    topk_ids,
    renormalize,
    expert_location_dispatch_info,
    num_token_non_padded,
):
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def npu_topk(
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        world_size: int,
        n_routed_experts: int,
        global_rank: int,
        correction_bias: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = 0,
        num_expert_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
):
    # TODO bias: currently not support optional
    # batch_size, seq_len, hidden_dim = hidden_states.shape
    batch_size, seq_len = hidden_states.shape
    topk_weight, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
        gating_output,
        k=topk,
        bias=torch.zeros(n_routed_experts, device="npu", dtype=gating_output.dtype),
        k_group=topk_group,
        group_count=num_expert_group,
        group_select_mode=1,
        renorm=0,
        norm_type=1,
        out_flag=False,
        routed_scaling_factor=routed_scaling_factor,
        eps=float(1e-20))
    step = batch_size // world_size * topk
    # TODO: experts_tp_size should set to correct value
    experts_tp_size = 1
    cur_topk_list = [
        (i + global_rank // experts_tp_size) % n_routed_experts for i in range(
            global_rank // experts_tp_size * step, (global_rank // experts_tp_size + 1) * step)]
    topk_ids = torch.Tensor(cur_topk_list).int().view(batch_size // world_size, -1).npu()
    return topk_weight, topk_ids

# This is used by the Deepseek V2/V3/R1 series models
# @torch.compile(dynamic=True, backend=get_compiler_backend())
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    log_info_on_rank0(logger, f"grouped_topk part 1 begin")
    scores = torch.softmax(gating_output, dim=-1)
    if _is_npu:
        scores = scores.to(torch.float32)
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    log_info_on_rank0(logger, f"grouped_topk part 2 begin")
    group_scores = (
        scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    log_info_on_rank0(logger, f"grouped_topk part 3 begin")
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    log_info_on_rank0(logger, f"grouped_topk part 4 begin")
    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
    topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    log_info_on_rank0(logger, f"grouped_topk part 5 begin")
    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor
    log_info_on_rank0(logger, f"grouped_topk part 6 begin")

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum

    log_info_on_rank0(logger, f"grouped_topk part 7 begin")
    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def biased_grouped_topk_impl(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"

    scores = gating_output.sigmoid()
    num_token = scores.shape[0]
    num_experts = scores.shape[1]
    scores_for_choice = scores.view(num_token, -1) + correction_bias.unsqueeze(0)
    group_scores = (
        scores_for_choice.view(num_token, num_expert_group, -1)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )  # [n, n_group]
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.shape[-1] // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores_for_choice.masked_fill(
        ~score_mask.bool(), float("-inf")
    )  # [n, e]
    _, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)
    topk_weights = scores.gather(1, topk_ids)

    if num_fused_shared_experts:
        topk_ids[:, -1] = torch.randint(
            low=num_experts,
            high=num_experts + num_fused_shared_experts,
            size=(topk_ids.size(0),),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        topk_weights[:, -1] = topk_weights[:, :-1].sum(dim=-1) / routed_scaling_factor

    if renormalize:
        topk_weights_sum = (
            topk_weights.sum(dim=-1, keepdim=True)
            if num_fused_shared_experts == 0
            else topk_weights[:, :-1].sum(dim=-1, keepdim=True)
        )
        topk_weights = topk_weights / topk_weights_sum

    topk_weights, topk_ids = topk_weights.to(torch.float32), topk_ids.to(torch.int32)
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_weights, topk_ids


def is_power_of_two(n):
    return n > 0 and math.log2(n).is_integer()


def _mask_topk_ids_padded_region(
    topk_ids: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    if num_token_non_padded is None:
        return
    indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
    topk_ids[indices >= num_token_non_padded, :] = -1


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _biased_grouped_topk_postprocess(
    topk_ids, expert_location_dispatch_info, num_token_non_padded
):
    topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)
    _mask_topk_ids_padded_region(topk_ids, num_token_non_padded)
    return topk_ids


def biased_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    compiled: bool = True,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
):
    assert (
        routed_scaling_factor is not None
    ), "routed_scaling_factor is required for biased_grouped_topk"
    # TODO: moe_fused_gate kernel is not supported for num_fused_shared_experts > 0 now.
    if (
        _is_cuda
        and gating_output.shape[1] // num_expert_group
        <= 32  # moe_fused_gate kernel ensure that num_experts/num_expert_group does not exceed MAX_VPT=32 now. And when kernel can handle MAX_VPT > 32, we can remove this assertion.
        and is_power_of_two(correction_bias.shape[0])
    ):
        topk_weights, topk_ids = moe_fused_gate(
            gating_output,
            correction_bias,
            num_expert_group,
            topk_group,
            topk,
            num_fused_shared_experts,
            routed_scaling_factor,
        )
        # TODO merge into kernel
        if (expert_location_dispatch_info is not None) or (
            num_token_non_padded is not None
        ):
            topk_ids = _biased_grouped_topk_postprocess(
                topk_ids, expert_location_dispatch_info, num_token_non_padded
            )
        return topk_weights, topk_ids
    else:
        biased_grouped_topk_fn = (
            torch.compile(
                biased_grouped_topk_impl, dynamic=True, backend=get_compiler_backend()
            )
            if compiled
            else biased_grouped_topk_impl
        )
        return biased_grouped_topk_fn(
            hidden_states,
            gating_output,
            correction_bias,
            topk,
            renormalize,
            num_expert_group,
            topk_group,
            num_fused_shared_experts=num_fused_shared_experts,
            routed_scaling_factor=routed_scaling_factor,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    num_fused_shared_experts: int = 0,
    custom_routing_function: Optional[Callable] = None,
    correction_bias: Optional[torch.Tensor] = None,
    torch_native: bool = False,
    routed_scaling_factor: Optional[float] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info: Optional[ExpertLocationDispatchInfo] = None,
    n_routed_experts: Optional[int] = None,
):
    router_logits, correction_bias = (
        expert_location_dispatch.transform_select_experts_inputs(
            router_logits=router_logits,
            correction_bias=correction_bias,
            info=expert_location_dispatch_info,
        )
    )
    log_info_on_rank0(logger, f"transform_select_experts_inputs end")
    # DeepSeek V2/V3/R1 series models use grouped_top_k
    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None
        if correction_bias is None:
            log_info_on_rank0(logger, f"correction_bias is None , grouped_topk start")
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
            log_info_on_rank0(logger, f"correction_bias is None , grouped_topk end")
        else:
            log_info_on_rank0(logger, f"correction_bias is not None , grouped_topk start")
            topk_weights, topk_ids = biased_grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                correction_bias=correction_bias,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                num_fused_shared_experts=num_fused_shared_experts,
                routed_scaling_factor=routed_scaling_factor,
                num_token_non_padded=num_token_non_padded,
                expert_location_dispatch_info=expert_location_dispatch_info,
            )
            log_info_on_rank0(logger, f"correction_bias is not None , grouped_topk end")
    elif torch_native and custom_routing_function is None:
        assert (
            num_token_non_padded is None
        ), "num_token_non_padded is not yet supported in fused_topk_native"
        assert expert_location_dispatch_info is None
        topk_weights, topk_ids = fused_topk_native(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
        )
    elif custom_routing_function is None:
        # Qwen3MOE uses fused_topk
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )
    else:
        # assert (
        #     num_token_non_padded is None
        # ), "num_token_non_padded is not yet supported in custom_routing_function"
        # assert expert_location_dispatch_info is None
        topk_weights, topk_ids = custom_routing_function(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=top_k,
            renormalize=renormalize,
            correction_bias=correction_bias,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            routed_scaling_factor=routed_scaling_factor,
            world_size=get_tensor_model_parallel_world_size(),
            n_routed_experts=n_routed_experts,
            global_rank=get_tensor_model_parallel_rank(),
        )
    log_info_on_rank0(logger, f"get_global_expert_distribution_recorder begin")
    get_global_expert_distribution_recorder().on_select_experts(topk_ids=topk_ids)
    log_info_on_rank0(logger, f"get_global_expert_distribution_recorder end")
    return topk_weights, topk_ids
