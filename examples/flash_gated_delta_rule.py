import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional

# Large default smoke shapes can exceed Triton-NPU's default launch-grid limit.
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu

from fla.ops.triton.triton_core.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum
from fla.ops.triton.triton_core.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.triton.triton_core.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard


_disable_compile = getattr(getattr(torch, "compiler", None), "disable", lambda fn: fn)
_DEFAULT_VARLEN_CHUNK_SIZES = (16, 32, 64, 128, 608 * 2)


def _make_gate(shape: tuple[int, ...], dtype: torch.dtype, device: str, gate_function: str) -> torch.Tensor:
    if gate_function == "logsigmoid":
        return F.logsigmoid(torch.randn(*shape, dtype=dtype, device=device))
    if gate_function == "negative_linear":
        lo, hi = -5e-2, -5e-5
        steps = shape[1] if len(shape) >= 2 else shape[-1]
        g_t = torch.linspace(hi, lo, steps, dtype=torch.float32, device=device).to(dtype)
        if len(shape) == 3:
            return g_t.view(1, steps, 1).expand(*shape).contiguous()
        return g_t.view(1, 1, steps, 1).expand(*shape).contiguous()
    if gate_function == "zeros":
        return torch.zeros(*shape, dtype=dtype, device=device)
    raise ValueError(f"Unsupported gate_function: {gate_function}")


def cdiv_torch(a, b):
    return (a + b - 1) // b


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor:
    indices = torch.cat(
        [torch.arange(n) for n in cdiv_torch(prepare_lens(cu_seqlens), chunk_size).tolist()]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_indices_list(cu_seqlens: list[int] | torch.LongTensor, chunk_size: int) -> list[int]:
    if isinstance(cu_seqlens, torch.Tensor):
        cu_seqlens = [int(x) for x in cu_seqlens.detach().cpu().tolist()]

    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = int(cu_seqlens[seq_idx + 1]) - int(cu_seqlens[seq_idx])
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def _as_int_list(value: Optional[list[int] | torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _activation_mode(activation: Optional[str]) -> int:
    if activation is None or activation == "":
        return 0
    if activation in ("silu", "swish"):
        return 1
    raise ValueError(f"Unsupported causal_conv1d activation: {activation}")


def _silu_backward(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return grad * sigmoid * (1.0 + x * (1.0 - sigmoid))


def _flatten_varlen_x(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[list[int]], bool]:
    if cu_seqlens is None:
        return x, None, False

    cu_list = _as_int_list(cu_seqlens)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError("causal_conv1d varlen path expects x.shape[0] == 1 for [1, T, D] input.")
        return x.reshape(x.shape[1], x.shape[2]).contiguous(), cu_list, True
    if x.ndim == 2:
        return x.contiguous(), cu_list, True
    raise ValueError(f"causal_conv1d varlen path expects rank-2 or rank-3 input, got shape={tuple(x.shape)}.")


def _flat_to_head_layout(x: torch.Tensor, head_num: int, *, is_varlen: bool) -> torch.Tensor:
    if head_num <= 0:
        return x
    if x.shape[-1] % head_num != 0:
        raise ValueError(f"last dimension must be divisible by head_num, got shape={tuple(x.shape)}, head_num={head_num}.")
    head_dim = x.shape[-1] // head_num
    if is_varlen:
        flat_x = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
        return flat_x.reshape(flat_x.shape[0], head_num, head_dim).transpose(0, 1).contiguous()
    return x.reshape(x.shape[0], x.shape[1], head_num, head_dim).transpose(1, 2).contiguous()


def _head_to_flat_layout(x: torch.Tensor, *, is_varlen: bool, batch_size: int) -> torch.Tensor:
    if is_varlen:
        x_head = x.squeeze(0) if x.ndim == 4 and x.shape[0] == 1 else x
        return x_head.transpose(0, 1).reshape(-1, x_head.shape[0] * x_head.shape[-1]).contiguous()
    return x.transpose(1, 2).reshape(batch_size, x.shape[2], x.shape[1] * x.shape[-1]).contiguous()


def _prepare_conv_states(
    x: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    num_sequences: int,
    width: int,
    dim: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    state_len = width - 1
    if initial_state is None:
        conv_states = torch.zeros(num_sequences, state_len, dim, dtype=x.dtype, device=x.device)
        return conv_states, None

    if initial_state.ndim != 3 or initial_state.shape[0] != num_sequences:
        raise ValueError(
            "initial_state must be rank-3 and match the sequence count, "
            f"got shape={tuple(initial_state.shape)} and num_sequences={num_sequences}."
        )

    if initial_state.shape[1] == dim and initial_state.shape[2] >= width:
        state_for_bwd = initial_state.transpose(1, 2).contiguous()
    elif initial_state.shape[2] == dim and initial_state.shape[1] >= width:
        state_for_bwd = initial_state.contiguous()
    else:
        raise ValueError(
            "initial_state must use [N, D, W] or [N, W, D] layout with W >= kernel width; "
            f"got shape={tuple(initial_state.shape)}, dim={dim}, width={width}."
        )

    conv_states = state_for_bwd[:, -state_len:, :].contiguous()
    return conv_states, state_for_bwd


def _causal_conv1d_final_state(
    x: torch.Tensor,
    *,
    width: int,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    dim = x.shape[-1]
    cu_list = _as_int_list(cu_seqlens)
    if cu_list is None:
        sequences = [x[i] for i in range(x.shape[0])]
    else:
        flat_x = x.reshape(-1, dim) if x.ndim == 3 else x
        sequences = [flat_x[cu_list[i] : cu_list[i + 1]] for i in range(len(cu_list) - 1)]

    states = []
    for idx, seq in enumerate(sequences):
        prev = None
        if initial_state is not None:
            prev = initial_state[idx]
            if prev.shape[0] != dim:
                prev = prev.transpose(0, 1).contiguous()
        hist = seq.transpose(0, 1).contiguous()
        if prev is not None:
            hist = torch.cat([prev[:, -width:], hist], dim=-1)
        if hist.shape[-1] < width:
            hist = F.pad(hist, (width - hist.shape[-1], 0))
        states.append(hist[:, -width:])
    return torch.stack(states, dim=0)


def solve_tri_ascendc(
    A: torch.Tensor,
    *,
    cu_seqlens: Optional[list[int] | torch.Tensor] = None,
    chunk_indices: Optional[list[int] | torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    if not hasattr(torch.ops.npu, "npu_solve_tri"):
        raise RuntimeError("torch.ops.npu.npu_solve_tri is unavailable. Please rebuild and install the latest fla_npu.")

    A_in = A.to(output_dtype).contiguous()
    cu_list = _as_int_list(cu_seqlens)
    chunk_list = _as_int_list(chunk_indices)

    if cu_list is None:
        return torch.ops.npu.npu_solve_tri(A_in, layout="bsnd")

    if A_in.ndim != 4 or A_in.shape[0] != 1:
        raise ValueError(f"solve_tri varlen path expects A with shape [1, T, H, BT], got {tuple(A_in.shape)}.")
    if chunk_list is None:
        raise ValueError("solve_tri varlen path requires chunk_indices.")

    out = torch.ops.npu.npu_solve_tri(
        A_in.squeeze(0),
        cu_seqlens=cu_list,
        chunk_indices=chunk_list,
        layout="tnd",
    )
    return out.unsqueeze(0)


class AscendCCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ):
        if not hasattr(torch.ops.npu, "npu_causal_conv1d"):
            raise RuntimeError("torch.ops.npu.npu_causal_conv1d is unavailable. Please rebuild and install fla_npu.")

        activation_mode = _activation_mode(activation)
        op_weight = weight.transpose(-1, -2).contiguous()
        width, dim = op_weight.shape
        op_x, query_start_loc, is_varlen = _flatten_varlen_x(x, cu_seqlens)
        num_sequences = len(query_start_loc) - 1 if query_start_loc is not None else int(x.shape[0])
        conv_states, initial_state_bwd = _prepare_conv_states(
            x,
            initial_state,
            num_sequences=num_sequences,
            width=width,
            dim=dim,
        )
        initial_state_mode = [1] * num_sequences if initial_state is not None else None

        preactivation = torch.ops.npu.npu_causal_conv1d(
            op_x,
            op_weight,
            bias,
            conv_states,
            query_start_loc=query_start_loc,
            initial_state_mode=initial_state_mode,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=H,
        )
        if is_varlen:
            preactivation = preactivation.unsqueeze(0)
        if residual is not None:
            preactivation = preactivation + _flat_to_head_layout(residual, H, is_varlen=is_varlen)

        y = F.silu(preactivation) if activation_mode != 0 else preactivation
        final_state = None
        if output_final_state:
            final_state = _causal_conv1d_final_state(
                x,
                width=width,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )

        ctx.save_for_backward(x, op_weight, bias, residual, initial_state_bwd, preactivation)
        ctx.activation_mode = activation_mode
        ctx.query_start_loc = query_start_loc
        ctx.is_varlen = is_varlen
        ctx.head_num = H
        ctx.batch_size = x.shape[0] if x.ndim == 3 else 1
        ctx.had_bias = bias is not None
        ctx.had_residual = residual is not None
        ctx.had_initial_state = initial_state is not None
        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        if not hasattr(torch.ops.npu, "npu_causal_conv1d_bwd"):
            raise RuntimeError("torch.ops.npu.npu_causal_conv1d_bwd is unavailable. Please rebuild and install fla_npu.")

        x, op_weight, bias, residual, initial_state_bwd, preactivation = ctx.saved_tensors
        op_x = x.reshape(-1, x.shape[-1]).contiguous() if ctx.is_varlen and x.ndim == 3 else x.contiguous()
        op_dy = dy.squeeze(0).contiguous() if ctx.is_varlen and dy.ndim == 4 else dy.contiguous()
        op_y = preactivation.squeeze(0).contiguous() if ctx.is_varlen and preactivation.ndim == 4 else preactivation.contiguous()
        dht_bwd = None
        if dht is not None:
            dht_bwd = dht.transpose(1, 2).contiguous() if dht.ndim == 3 and dht.shape[1] == x.shape[-1] else dht

        dx, dw, db, dh0 = torch.ops.npu.npu_causal_conv1d_bwd(
            x=op_x,
            y=op_y if ctx.activation_mode != 0 else None,
            weight=op_weight,
            dy=op_dy,
            initial_state=initial_state_bwd if ctx.had_initial_state else None,
            dht=dht_bwd,
            query_start_loc=ctx.query_start_loc,
            activation=ctx.activation_mode,
            input_layout="NTD" if ctx.is_varlen else "BNSD",
        )

        dx = dx.reshape_as(x)
        dw = dw.transpose(0, 1).contiguous()
        db = db if ctx.had_bias else None
        dr = None
        if ctx.had_residual:
            dr_head = _silu_backward(dy, preactivation) if ctx.activation_mode != 0 else dy
            dr = _head_to_flat_layout(dr_head, is_varlen=ctx.is_varlen, batch_size=ctx.batch_size)
        dh0 = dh0.transpose(1, 2).contiguous() if ctx.had_initial_state else None
        return dx, dw, None, db, dr, dh0, None, None, None


def causal_conv1d_ascendc(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    return AscendCCausalConv1dFunction.apply(
        x,
        weight,
        H,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )


def _as_chunk_dict(
    value: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_size: int,
) -> Dict[str, Optional[torch.LongTensor]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {str(chunk_size): value}


def _as_chunk_list_dict(
    value: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> Dict[str, Optional[list[int]]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): _as_int_list(v) for k, v in value.items()}
    return {str(chunk_size): _as_int_list(value)}


def _next_power_of_2(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _cumsum_block_t(g: torch.Tensor, chunk_size: int) -> int:
    # Keep this aligned with fla.ops.triton.triton_core.cumsum.chunk_local_cumsum_scalar.
    h = int(g.shape[-1])
    return _next_power_of_2((1 << 17) // max(1, h * int(chunk_size)))


def _ensure_varlen_metadata(
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    cu_seqlens_list: Optional[list[int]],
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor],
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor],
    chunk_size: int,
) -> tuple[
    Optional[torch.LongTensor],
    Optional[list[int]],
    Optional[Dict[str, Optional[torch.LongTensor]]],
    Optional[Dict[str, Optional[list[int]]]],
]:
    if cu_seqlens is None:
        return None, None, None, None

    cu_seqlens = cu_seqlens.to(device=g.device, dtype=torch.int64)
    cu_seqlens_list = _as_int_list(cu_seqlens_list) or _as_int_list(cu_seqlens)
    assert cu_seqlens_list is not None

    tensor_indices = _as_chunk_dict(chunk_indices, chunk_size)
    list_indices = _as_chunk_list_dict(chunk_indices_list, chunk_size)

    required_sizes = set(_DEFAULT_VARLEN_CHUNK_SIZES)
    required_sizes.add(int(chunk_size))
    required_sizes.add(_cumsum_block_t(g, chunk_size))

    for size in required_sizes:
        key = str(size)
        if key not in tensor_indices or tensor_indices[key] is None:
            tensor_indices[key] = prepare_chunk_indices(cu_seqlens, size)
        if key not in list_indices or list_indices[key] is None:
            list_indices[key] = prepare_chunk_indices_list(cu_seqlens_list, size)

    return cu_seqlens, cu_seqlens_list, tensor_indices, list_indices


def _chunk_tensor(
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]],
    chunk_size: int,
) -> Optional[torch.LongTensor]:
    if chunk_indices is None:
        return None
    return chunk_indices.get(str(chunk_size))


def _chunk_list(
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]],
    chunk_size: int,
) -> Optional[list[int]]:
    if chunk_indices_list is None:
        return None
    return chunk_indices_list.get(str(chunk_size))


_RECOMPUTE_WU_FALLBACK_WARNED = False
_RECOMPUTE_WU_FORCE_FALLBACK = False


def _pad_time_dim(x: torch.Tensor, pad_t: int) -> torch.Tensor:
    if pad_t == 0:
        return x
    if x.ndim == 3:
        return F.pad(x, (0, pad_t))
    if x.ndim == 4:
        return F.pad(x, (0, 0, 0, pad_t))
    raise ValueError(f"expected rank-3 or rank-4 tensor, got shape={tuple(x.shape)}.")


def _recompute_w_u_torch(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    beta_k = beta.to(k.dtype)
    exp_g = torch.exp(g.float()).to(k.dtype)
    k_scaled = k * (beta_k * exp_g).unsqueeze(-1)
    v_scaled = v * beta_k.unsqueeze(-1)

    if cu_seqlens is None:
        B, H, T, K = k.shape
        V = v.shape[-1]
        num_chunks = (T + chunk_size - 1) // chunk_size
        padded_t = num_chunks * chunk_size
        pad_t = padded_t - T
        A_blocks = _pad_time_dim(A, pad_t).reshape(B, H, num_chunks, chunk_size, chunk_size)
        k_blocks = _pad_time_dim(k_scaled, pad_t).reshape(B, H, num_chunks, chunk_size, K)
        v_blocks = _pad_time_dim(v_scaled, pad_t).reshape(B, H, num_chunks, chunk_size, V)
        w = torch.matmul(A_blocks, k_blocks).reshape(B, H, padded_t, K)[:, :, :T, :].contiguous()
        u = torch.matmul(A_blocks, v_blocks).reshape(B, H, padded_t, V)[:, :, :T, :].contiguous()
        return w.to(k.dtype), u.to(v.dtype)

    if chunk_indices is None:
        raise ValueError("chunk_indices is required when cu_seqlens is provided.")

    w = torch.empty_like(k)
    u = torch.empty_like(v)
    for idx in range(0, len(chunk_indices), 2):
        seq_idx = chunk_indices[idx]
        chunk_idx = chunk_indices[idx + 1]
        bos = cu_seqlens[seq_idx] + chunk_idx * chunk_size
        eos = min(bos + chunk_size, cu_seqlens[seq_idx + 1])
        if eos <= bos:
            continue
        A_chunk = A[:, :, bos:eos, : eos - bos]
        w[:, :, bos:eos, :] = torch.matmul(A_chunk, k_scaled[:, :, bos:eos, :]).to(k.dtype)
        u[:, :, bos:eos, :] = torch.matmul(A_chunk, v_scaled[:, :, bos:eos, :]).to(v.dtype)
    return w, u


def recompute_w_u(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    *,
    chunk_size: int,
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    global _RECOMPUTE_WU_FALLBACK_WARNED, _RECOMPUTE_WU_FORCE_FALLBACK

    use_recompute_op = os.environ.get("FLASH_GDR_USE_ASCENDC_RECOMPUTE", "0") == "1"
    if use_recompute_op and not _RECOMPUTE_WU_FORCE_FALLBACK and hasattr(torch.ops.npu, "npu_recompute_w_u_fwd"):
        try:
            w, u = torch.ops.npu.npu_recompute_w_u_fwd(
                k,
                v,
                beta,
                A,
                chunk_size,
                g=g,
                gk=None,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            torch.npu.synchronize()
            return w, u
        except RuntimeError as exc:
            message = str(exc)
            if "RecomputeWUFwd" not in message and "npu_recompute_w_u_fwd" not in message:
                raise
            _RECOMPUTE_WU_FORCE_FALLBACK = True
            if not _RECOMPUTE_WU_FALLBACK_WARNED:
                warnings.warn(
                    "npu_recompute_w_u_fwd failed to launch; falling back to torch chunk matmul for this example.",
                    RuntimeWarning,
                )
                _RECOMPUTE_WU_FALLBACK_WARNED = True

    return _recompute_w_u_torch(
        k,
        v,
        beta,
        A,
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


def flash_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    # A is the WY lower-triangular representation before inversion.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )

    A = solve_tri_ascendc(
        A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        output_dtype=k.dtype,
    )

    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()
    A = A.transpose(1, 2).contiguous()

    w, u = recompute_w_u(
        k,
        v,
        beta,
        A,
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    if not output_final_state:
        final_state = None

    o = torch.ops.npu.npu_chunk_fwd_o(
        q,
        k,
        v_new,
        h,
        scale,
        g=g,
        g_gamma=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        chunk_size=chunk_size,
        transpose_state_layout=False,
    )

    g = g.transpose(1, 2).contiguous()
    o = o.transpose(1, 2).contiguous()
    return g, o, A, final_state


def flash_chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    do: torch.Tensor,
    dht: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
    chunk_size: int = 64,
):
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous().float()

    w, u = recompute_w_u(
        k,
        v,
        beta,
        A,
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    do = do.transpose(1, 2).contiguous()

    h, v_new, _ = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        g=g,
        gk=None,
        initial_state=initial_state,
        output_final_state=False,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )

    dv = torch.ops.npu.npu_chunk_bwd_dv_local(
        q,
        k,
        do,
        g,
        scale,
        chunk_size,
        g_gamma=None,
        A=A,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dh, dh0, dv = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        q,
        k,
        w,
        do,
        dv,
        scale,
        chunk_size,
        g=g,
        gK=None,
        h0=None,
        dht=None,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        use_exp2=False,
        transpose_state_layout=False,
    )
    dh0 = None

    dq, dk, dw, dg = torch.ops.npu.npu_chunk_bwd_dqkwg(
        q,
        k,
        v_new,
        g,
        h,
        do,
        dh,
        dv,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
        w=None,
        g_gamma=None,
        scale=scale,
        use_exp2=False,
        transpose_state_layout=False,
    )

    dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k,
        v,
        beta.float(),
        A,
        dw,
        dv,
        g.float(),
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    dk2, dv, db, dg2 = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k,
        v,
        beta,
        A,
        dA,
        dw,
        dv,
        g,
        chunk_size,
        cu_seqlens=cu_seqlens_list,
        chunk_indices=_chunk_list(chunk_indices_list, chunk_size),
    )

    db = db.transpose(1, 2).contiguous()
    dg2 = dg2.transpose(1, 2).contiguous()
    dg = dg.transpose(1, 2).contiguous()

    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(f"dg current type is {dg.dtype}, should be float32")

    dg = chunk_local_cumsum(
        dg,
        chunk_size=chunk_size,
        reverse=True,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices,
        head_first=False,
    )

    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        cu_seqlens_list: Optional[list[int]] = None,
        chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]]] = None,
        chunk_indices_list: Optional[Dict[str, Optional[list[int]]]] = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ):
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)
        else:
            q_rstd, k_rstd = None, None

        g, o, A, final_state = flash_chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )

        ctx.save_for_backward(q, k, v, g, beta, A)
        ctx.q_rstd = q_rstd
        ctx.k_rstd = k_rstd
        ctx.initial_state = initial_state
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cu_seqlens_list = cu_seqlens_list
        ctx.chunk_indices = chunk_indices
        ctx.chunk_indices_list = chunk_indices_list
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):
        q, k, v, g, beta, A = ctx.saved_tensors
        dq, dk, dv, db, dg, dh0 = flash_chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=ctx.initial_state,
            do=do,
            dht=dht,
            cu_seqlens=ctx.cu_seqlens,
            cu_seqlens_list=ctx.cu_seqlens_list,
            chunk_indices=ctx.chunk_indices,
            chunk_indices_list=ctx.chunk_indices_list,
            chunk_size=ctx.chunk_size,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, ctx.q_rstd, dq)
            dk = l2norm_bwd(k, ctx.k_rstd, dk)
        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g),
            db.to(beta),
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@_disable_compile
def flash_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    cu_seqlens_list: Optional[list[int]] = None,
    chunk_indices: Optional[Dict[str, Optional[torch.LongTensor]] | torch.LongTensor] = None,
    chunk_indices_list: Optional[Dict[str, Optional[list[int]]] | list[int] | torch.Tensor] = None,
    chunk_size: int = 64,
    head_first: bool = False,
):
    r"""
    Flash-linear-attention NPU port of xtuner's GDN entry.

    This port keeps the xtuner NPU layout:
        q, k: [B, H, T, K]
        v:    [B, H, T, V]
        g:    [B, T, H]
        beta: [B, T, H]

    It returns:
        o: [B, T, H, V]
        final_state: [N, H, K, V] when output_final_state=True, else None
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype}, k current type is {k.dtype}, "
            f"v current type is {v.dtype}; they should be equal"
        )
    if q.dtype == torch.float32:
        raise ValueError("ChunkGatedDeltaRuleFunction does not support float32. Please use float16/bfloat16.")
    if beta.ndim != 3 or g.ndim != 3:
        raise ValueError("g and beta must be rank-3 tensors with shape [B, T, H].")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be rank-4 tensors with shape [B, H, T, D].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"q/k/v shape prefixes must match, got {q.shape}, {k.shape}, {v.shape}.")
    if g.shape != beta.shape:
        raise ValueError(f"g and beta shapes must match, got {g.shape} and {beta.shape}.")
    if g.shape[0] != q.shape[0] or g.shape[1] != q.shape[2] or g.shape[2] != q.shape[1]:
        raise ValueError(
            "Expected q/k/v in [B, H, T, D] and g/beta in [B, T, H]; "
            f"got q={tuple(q.shape)}, g={tuple(g.shape)}."
        )

    if head_first:
        warnings.warn(
            "head_first is kept only for API compatibility. This NPU port always expects q/k/v as [B, H, T, D].",
            stacklevel=2,
        )
    if chunk_size != 2 ** (chunk_size.bit_length() - 1):
        raise ValueError(f"chunk_size must be a power of 2, got {chunk_size}.")

    if cu_seqlens is not None:
        cu_seqlens, cu_seqlens_list, chunk_indices, chunk_indices_list = _ensure_varlen_metadata(
            g=g,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=chunk_indices,
            chunk_indices_list=chunk_indices_list,
            chunk_size=chunk_size,
        )
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using cu_seqlens. "
                "Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens_list) - 1:
            raise ValueError(
                "The number of initial states is expected to match the number of input sequences, "
                f"got initial_state.shape[0]={initial_state.shape[0]} and sequences={len(cu_seqlens_list) - 1}."
            )
    else:
        cu_seqlens_list = None
        chunk_indices = None
        chunk_indices_list = None

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        initial_state,
        output_final_state,
        cu_seqlens,
        cu_seqlens_list,
        chunk_indices,
        chunk_indices_list,
        use_qk_l2norm_in_kernel,
        chunk_size,
    )
    return o, final_state


class QwenStyleRMSNormGated(nn.Module):
    """RMSNorm + gated SiLU."""

    def __init__(self, head_v_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_v_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        inp_dtype = hidden_states.dtype
        hidden_states = torch_npu.npu_rms_norm(hidden_states, weight, self.variance_epsilon)[0]
        hidden_states = hidden_states * torch.nn.functional.silu(gate.to(hidden_states.dtype))
        return hidden_states.to(inp_dtype)


class DemoGatedDeltaNet(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        *,
        num_value_heads: int,
        num_key_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_dim: int,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_v_heads = num_value_heads
        self.num_k_heads = num_key_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError(
                "num_value_heads must be an integer multiple of num_key_heads "
                f"for the current grouped-value smoke path, got {self.num_v_heads} and {self.num_k_heads}."
            )
        if key_head_dim != value_head_dim:
            raise ValueError(
                "DemoGatedDeltaNet uses causal_conv1d head-first output, which requires "
                f"key_head_dim == value_head_dim, got {key_head_dim} and {value_head_dim}."
            )
        self.head_k_dim = key_head_dim
        self.head_v_dim = value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = conv_kernel_dim
        self.activation = hidden_act
        self.chunk_size_default = chunk_size

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.rand(self.num_v_heads) * 15.99 + 0.01))
        self.in_proj_qkv = nn.Linear(self.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.norm = QwenStyleRMSNormGated(self.head_v_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: Optional[torch.LongTensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if batch_size != 1:
            raise ValueError("DemoGatedDeltaNet 仅对齐 xtuner bs=1 路径")

        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        weight = self.conv1d.weight.squeeze(1)
        bias = self.conv1d.bias

        if cu_seqlens is not None and cu_seqlens.device != mixed_qkv.device:
            cu_seqlens = cu_seqlens.to(mixed_qkv.device)

        mixed_qkv, _ = causal_conv1d_ascendc(
            mixed_qkv,
            weight=weight,
            H=2 * self.num_k_heads + self.num_v_heads,
            bias=bias,
            residual=None,
            initial_state=None,
            activation=self.activation,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )

        query = mixed_qkv[:, : self.num_k_heads].contiguous()
        key = mixed_qkv[:, self.num_k_heads : 2 * self.num_k_heads].contiguous()
        value = mixed_qkv[:, 2 * self.num_k_heads :].contiguous()

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * torch.nn.functional.softplus(a.float() + self.dt_bias)

        repeat = self.num_v_heads // self.num_k_heads
        if repeat > 1:
            query = query.repeat_interleave(repeat, dim=1)
            key = key.repeat_interleave(repeat, dim=1)

        cu_list = cu_seqlens.detach().tolist() if cu_seqlens is not None else None
        cu_seqlens_list: Optional[list[int]] = cu_list if cu_list is not None else None

        core_attn_out, _ = flash_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_list=cu_seqlens_list,
            chunk_indices=None,
            chunk_indices_list=None,
            chunk_size=self.chunk_size_default,
        )

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z_flat)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)


__all__ = [
    "flash_gated_delta_rule",
    "flash_chunk_gated_delta_rule_fwd",
    "flash_chunk_gated_delta_rule_bwd",
    "prepare_chunk_indices",
    "prepare_chunk_indices_list",
    "DemoGatedDeltaNet",
    "QwenStyleRMSNormGated",
]


def _build_mean_1k_cu_seqlens(
    total_tokens: int,
    chunk_size: int,
    device: str,
    mean_len: int = 1024,
) -> torch.LongTensor:
    num_seqs = max(1, round(total_tokens / mean_len))
    if num_seqs == 1:
        lengths = [total_tokens]
    else:
        target = max(1, total_tokens // num_seqs)
        delta = max(chunk_size, (target // 4 // chunk_size) * chunk_size)
        low = max(1, target - delta)
        high = max(1, target + delta)
        lengths = [low if i % 2 == 0 else high for i in range(num_seqs)]

        diff = total_tokens - sum(lengths)
        i = len(lengths) - 1
        while diff != 0:
            if diff > 0:
                add = min(diff, max(1, delta))
                lengths[i] += add
                diff -= add
            else:
                sub = min(-diff, max(0, lengths[i] - 1))
                if sub == 0:
                    i = (i - 1) % len(lengths)
                    continue
                lengths[i] -= sub
                diff += sub
            i = (i - 1) % len(lengths)

    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return torch.tensor(offsets, dtype=torch.int64, device=device)


def _main():
    import argparse

    import torch_npu

    import fla_npu  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default="", help="Example/ST case name for CI logs")
    parser.add_argument("--device", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=32, help="legacy alias for --query-heads and --value-heads")
    parser.add_argument("--query-heads", type=int, default=None)
    parser.add_argument("--value-heads", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=65536)
    parser.add_argument("--dim", type=int, default=128, help="legacy alias for --key-dim")
    parser.add_argument("--key-dim", type=int, default=None)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--mean-len", type=int, default=1024)
    parser.add_argument("--gate-source", choices=["g", "gk", "g+gk"], default="g")
    parser.add_argument("--gate-function", choices=["logsigmoid", "negative_linear", "zeros"], default="logsigmoid")
    parser.add_argument("--initial-state", choices=["none", "zeros", "random"], default="none")
    parser.add_argument("--output-final-state", action="store_true")
    parser.add_argument("--qk-l2norm", dest="qk_l2norm", action="store_true", default=True)
    parser.add_argument("--no-qk-l2norm", dest="qk_l2norm", action="store_false")
    parser.add_argument("--varlen", dest="varlen", action="store_true", default=True)
    parser.add_argument("--no-varlen", dest="varlen", action="store_false")
    parser.add_argument(
        "--demo-model",
        action="store_true",
        help="运行 DemoGatedDeltaNet（从 cu_seqlens 迁移到 tensor 设备及 causal_conv 起始的完整链路）代替裸张量 attn 冒烟",
    )
    parser.add_argument("--conv-kernel", type=int, default=4, help="depthwise causal conv kernel size")
    args = parser.parse_args()

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    query_heads = args.query_heads if args.query_heads is not None else args.heads
    value_heads = args.value_heads if args.value_heads is not None else args.heads
    key_dim = args.key_dim if args.key_dim is not None else args.dim
    value_dim = args.value_dim
    batch = args.batch
    device = f"npu:{args.device}"
    positive_values = {
        "batch": batch,
        "tokens": args.tokens,
        "query_heads": query_heads,
        "value_heads": value_heads,
        "key_dim": key_dim,
        "value_dim": value_dim,
        "chunk_size": args.chunk_size,
        "mean_len": args.mean_len,
    }
    for name, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if args.varlen and batch != 1:
        raise ValueError("varlen smoke currently requires batch=1. Use --no-varlen for B > 1 cases.")
    if value_heads % query_heads != 0:
        raise ValueError(
            "value_heads must be an integer multiple of query_heads for the current grouped-value smoke path, "
            f"got query_heads={query_heads}, value_heads={value_heads}."
        )
    if args.gate_source != "g":
        raise NotImplementedError(
            f"gate_source={args.gate_source} is reserved in the Example/ST schema, but the current NPU "
            "fwd_h path only supports g and requires gk=None."
        )

    if args.demo_model:
        if (
            args.gate_source != "g"
            or args.gate_function != "logsigmoid"
            or args.initial_state != "none"
            or args.output_final_state
            or not args.qk_l2norm
        ):
            raise ValueError(
                "demo_model smoke currently uses its built-in gate, state and qk-l2norm path. "
                "Use demo_model=false for gate/initial_state/output_final_state/qk_l2norm case coverage."
            )
        if batch != 1:
            raise ValueError("demo_model smoke currently requires batch=1.")
        hidden_size = query_heads * key_dim
        if hidden_size <= 0:
            raise ValueError("hidden_size = query_heads * key_dim 非法")
        cu_seqlens = None
        if args.varlen:
            cu_seqlens = _build_mean_1k_cu_seqlens(
                total_tokens=args.tokens,
                chunk_size=args.chunk_size,
                device=device,
                mean_len=args.mean_len,
            )
        x = torch.randn(batch, args.tokens, hidden_size, dtype=dtype, device=device, requires_grad=True)
        net = DemoGatedDeltaNet(
            hidden_size,
            num_value_heads=value_heads,
            num_key_heads=query_heads,
            key_head_dim=key_dim,
            value_head_dim=value_dim,
            conv_kernel_dim=args.conv_kernel,
            chunk_size=args.chunk_size,
        ).to(device=device, dtype=dtype)
        torch.npu.synchronize()
        out = net(x, cu_seqlens=cu_seqlens)
        torch.npu.synchronize()
        print("demo_model forward:", tuple(out.shape), out.dtype, "finite:", torch.isfinite(out.float()).all().item())
        loss = out.float().square().mean()
        loss.backward()
        torch.npu.synchronize()
        print("demo_model backward ok")
        print(
            "x.grad finite:",
            torch.isfinite(x.grad.float()).all().item() if x.grad is not None else None,
            "norm:",
            float(x.grad.float().norm()) if x.grad is not None else None,
        )
        return

    q = torch.randn(batch, query_heads, args.tokens, key_dim, dtype=dtype, device=device)
    k = torch.randn(batch, query_heads, args.tokens, key_dim, dtype=dtype, device=device)
    v = torch.randn(batch, value_heads, args.tokens, value_dim, dtype=dtype, device=device)
    beta = torch.rand(batch, args.tokens, value_heads, dtype=dtype, device=device).sigmoid()
    g = _make_gate((batch, args.tokens, value_heads), dtype, device, args.gate_function)

    cu_seqlens = None
    if args.varlen:
        cu_seqlens = _build_mean_1k_cu_seqlens(
            total_tokens=args.tokens,
            chunk_size=args.chunk_size,
            device=device,
            mean_len=args.mean_len,
        )
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        print(
            "varlen:",
            "num_seqs=", int(lens.numel()),
            "mean_len=", float(lens.float().mean().item()),
            "min_len=", int(lens.min().item()),
            "max_len=", int(lens.max().item()),
        )

    print(
        "config:",
        f"case={args.case_name or '<direct>'}",
        f"B={batch}",
        f"QH={query_heads}",
        f"VH={value_heads}",
        f"T={args.tokens}",
        f"K={key_dim}",
        f"V={value_dim}",
        f"chunk_size={args.chunk_size}",
        f"dtype={args.dtype}",
        f"varlen={args.varlen}",
        f"gate_source={args.gate_source}",
        f"gate_function={args.gate_function}",
        f"initial_state={args.initial_state}",
        f"output_final_state={args.output_final_state}",
        f"qk_l2norm={args.qk_l2norm}",
        f"device={device}",
    )

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    beta.requires_grad_(True)
    g.requires_grad_(True)

    torch.npu.synchronize()
    attn_q = q
    attn_k = k
    if query_heads != value_heads:
        repeat = value_heads // query_heads
        attn_q = q.repeat_interleave(repeat, dim=1)
        attn_k = k.repeat_interleave(repeat, dim=1)
        print("grouped value heads:", f"repeat={repeat}", f"attn_heads={attn_q.shape[1]}")
    initial_state = None
    if args.initial_state != "none":
        state_count = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
        state_shape = (state_count, value_heads, key_dim, value_dim)
        if args.initial_state == "zeros":
            initial_state = torch.zeros(state_shape, dtype=dtype, device=device)
        else:
            initial_state = torch.randn(state_shape, dtype=dtype, device=device)
        print("initial_state:", tuple(initial_state.shape), initial_state.dtype, initial_state.device)
    o, final_state = flash_gated_delta_rule(
        attn_q,
        attn_k,
        v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=args.output_final_state,
        use_qk_l2norm_in_kernel=args.qk_l2norm,
        cu_seqlens=cu_seqlens,
        chunk_size=args.chunk_size,
    )
    torch.npu.synchronize()

    print("forward ok")
    print("o:", tuple(o.shape), o.dtype, o.device, "finite:", torch.isfinite(o.float()).all().item())
    print("final_state:", None if final_state is None else tuple(final_state.shape))

    loss = o.float().square().mean()
    loss.backward()
    torch.npu.synchronize()
    print("backward ok")
    for name, tensor in [("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)]:
        grad = tensor.grad
        print(name, "grad finite:", torch.isfinite(grad.float()).all().item(), "norm:", grad.float().norm().item())


if __name__ == "__main__":
    _main()
