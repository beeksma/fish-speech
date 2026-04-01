"""INT4 Polar-Quantized KV Cache Attention — Triton kernel for ROCm.

Phase 1 prototype: correct first, fast later.

Uses pre-rotated Q to avoid per-tile R^T multiply:
  score_j = (Q@R) · centroids[K_indices_j] * K_mag_j/√D + K_mean_j * sum(Q)
  O_rot = Σ attn_j * (centroids[V_indices_j] * V_mag_j/√D + V_mean_j)
  O = O_rot @ R^T
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _int4_attn_kernel(
    Q_rot_ptr,       # (num_q, D) bf16 — pre-rotated queries
    K_packed_ptr,    # (seq_kv, D//2) uint8 — packed 4-bit K
    K_mag_ptr,       # (seq_kv,) bf16
    K_mean_ptr,      # (seq_kv,) bf16
    V_packed_ptr,    # (seq_kv, D//2) uint8 — packed 4-bit V
    V_mag_ptr,       # (seq_kv,) bf16
    V_mean_ptr,      # (seq_kv,) bf16
    C_ptr,           # (16,) bf16 — centroids
    Q_sum_ptr,       # (num_q,) float32 — sum of original Q elements
    O_rot_ptr,       # (num_q, D) bf16 — output in rotated space
    seq_kv,          # int: filled KV positions
    stride_q,        # stride for Q rows
    stride_kp,       # stride for K_packed rows
    stride_vp,       # stride for V_packed rows
    stride_o,        # stride for O rows
    inv_sqrt_d: tl.constexpr,  # 1/sqrt(D)
    HEAD_DIM: tl.constexpr,
    PACKED_DIM: tl.constexpr,
):
    """One program per query position. Loops over KV positions sequentially.

    Simple and correct. Processes full head_dim vectors per KV position.
    """
    q_idx = tl.program_id(0)

    # Load centroids into registers
    c = tl.load(C_ptr + tl.arange(0, 16)).to(tl.float32)

    # Load query vector (rotated)
    d_offs = tl.arange(0, HEAD_DIM)
    q_rot = tl.load(Q_rot_ptr + q_idx * stride_q + d_offs).to(tl.float32)
    q_sum = tl.load(Q_sum_ptr + q_idx).to(tl.float32)

    # Online softmax state
    m_running = -1e30  # running max score
    d_running = 0.0    # running softmax denominator
    o_running = tl.zeros([HEAD_DIM], dtype=tl.float32)  # running output

    for j in range(seq_kv):
        # ── Dequantize K_j ──
        # Load packed byte for each dimension pair, unpack to centroid value
        # d_offs // 2 maps each head_dim index to its packed byte
        k_byte = tl.load(K_packed_ptr + j * stride_kp + d_offs // 2).to(tl.int32)
        # Even dims use low nibble, odd dims use high nibble
        k_idx = tl.where(d_offs % 2 == 0, k_byte & 0x0F, (k_byte >> 4) & 0x0F)
        k_rot_full = tl.load(C_ptr + k_idx).to(tl.float32)

        k_mag = tl.load(K_mag_ptr + j).to(tl.float32)
        k_mean = tl.load(K_mean_ptr + j).to(tl.float32)

        # Score: Q_rot · K_rot * mag/√D + mean * sum(Q)
        score = tl.sum(q_rot * k_rot_full) * k_mag * inv_sqrt_d + k_mean * q_sum
        score = score * inv_sqrt_d  # final 1/√D scaling for attention

        # ── Online softmax ──
        m_new = tl.maximum(m_running, score)
        exp_old = tl.exp(m_running - m_new)
        exp_new = tl.exp(score - m_new)
        d_new = d_running * exp_old + exp_new

        # ── Dequantize V_j ──
        v_byte = tl.load(V_packed_ptr + j * stride_vp + d_offs // 2).to(tl.int32)
        v_idx = tl.where(d_offs % 2 == 0, v_byte & 0x0F, (v_byte >> 4) & 0x0F)
        v_rot_full = tl.load(C_ptr + v_idx).to(tl.float32)

        v_mag = tl.load(V_mag_ptr + j).to(tl.float32)
        v_mean = tl.load(V_mean_ptr + j).to(tl.float32)

        # Reconstruct V: centroids * mag/√D + mean
        v_deq = v_rot_full * v_mag * inv_sqrt_d + v_mean

        # Accumulate weighted V with softmax rescaling
        o_running = o_running * exp_old + exp_new * v_deq

        m_running = m_new
        d_running = d_new

    # Normalize
    o_running = o_running / d_running

    # Store
    tl.store(O_rot_ptr + q_idx * stride_o + d_offs, o_running.to(tl.bfloat16))


def int4_polar_attention(
    q: torch.Tensor,          # (num_q, head_dim) bf16
    k_packed: torch.Tensor,   # (max_seq, packed_dim) uint8
    k_mag: torch.Tensor,      # (max_seq,) bf16
    k_mean: torch.Tensor,     # (max_seq,) bf16
    v_packed: torch.Tensor,   # (max_seq, packed_dim) uint8
    v_mag: torch.Tensor,      # (max_seq,) bf16
    v_mean: torch.Tensor,     # (max_seq,) bf16
    centroids: torch.Tensor,  # (16,) bf16
    rotation: torch.Tensor,   # (head_dim, head_dim) bf16
    seq_kv: int,
) -> torch.Tensor:
    """INT4 attention — computes attention directly on packed INT4 KV cache.

    Never materializes full bf16 KV tensors in global memory.
    """
    num_q, head_dim = q.shape
    packed_dim = head_dim // 2

    # Pre-rotate Q once: Q_rot = Q @ R
    q_rot = (q.float() @ rotation.float()).to(q.dtype)

    # Sum of original Q elements (for mean correction term)
    q_sum = q.float().sum(dim=-1)

    # Output buffer
    o_rot = torch.empty(num_q, head_dim, dtype=q.dtype, device=q.device)

    inv_sqrt_d = 1.0 / math.sqrt(head_dim)

    grid = (num_q,)
    _int4_attn_kernel[grid](
        q_rot,
        k_packed, k_mag, k_mean,
        v_packed, v_mag, v_mean,
        centroids, q_sum, o_rot,
        seq_kv=seq_kv,
        stride_q=q_rot.stride(0),
        stride_kp=k_packed.stride(0),
        stride_vp=v_packed.stride(0),
        stride_o=o_rot.stride(0),
        inv_sqrt_d=inv_sqrt_d,
        HEAD_DIM=head_dim,
        PACKED_DIM=packed_dim,
    )

    # Rotate output back: O = O_rot @ R^T
    output = (o_rot.float() @ rotation.float().T).to(q.dtype)
    return output
