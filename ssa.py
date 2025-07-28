import pytest
import torch

import triton
import triton.language as tl

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _fwd_kernel(
    Q1, Q2, K, V1, V2, sm_scale,
    L,
    Out,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_v1z, stride_v1h, stride_v1k, stride_v1n,
    stride_v2z, stride_v2h, stride_v2k, stride_v2n,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_q1h
    Q1_block_ptr = tl.make_block_ptr(
        base=Q1 + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_q1m, stride_q1k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    Q2_block_ptr = tl.make_block_ptr(
        base=Q2 + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_q2m, stride_q2k),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V1_block_ptr = tl.make_block_ptr(
        base=V1 + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_v1k, stride_v1n),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    V2_block_ptr = tl.make_block_ptr(
        base=V2 + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_v2k, stride_v2n),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q1 = tl.load(Q1_block_ptr)
    q1 = (q1 * qk_scale).to(tl.float16)
    q2 = tl.load(Q2_block_ptr)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v1 = tl.load(V1_block_ptr)
        v2 = tl.load(V2_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q1, k)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        S_ = v1[:, :, None] * v2[:, None, :]
        S_ = tl.sum(p.to(tl.float16)[:, :, None, None] *  S_[None, :, :, :], axis = 1)
        acc *= acc_scale[:, None]
        acc += tl.sum(q2[:, :, None] * S_, axis = 1)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V1_block_ptr = tl.advance(V1_block_ptr, (BLOCK_N, 0))
        V2_block_ptr = tl.advance(V2_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(tl.float16))


@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel(
    Q1, Q2, K, V1, V2, sm_scale, Out, DO,
    DQ1, DQ2, DK, DV1, DV2,
    L,
    D,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_v1z, stride_v1h, stride_v1k, stride_v1n,
    stride_v2z, stride_v2h, stride_v2k, stride_v2n,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504
    # offset pointers for batch/head
    Q1 += off_z * stride_q1z + off_h * stride_q1h
    Q2 += off_z * stride_q2z + off_h * stride_q2h
    K += off_z * stride_q1z + off_h * stride_q1h
    V1 += off_z * stride_q1z + off_h * stride_q1h
    V2 += off_z * stride_q2z + off_h * stride_q2h
    DO += off_z * stride_q2z + off_h * stride_q2h
    DQ1 += off_z * stride_q1z + off_h * stride_q1h
    DQ2 += off_z * stride_q2z + off_h * stride_q2h
    DK += off_z * stride_q1z + off_h * stride_q1h
    DV1 += off_z * stride_q1z + off_h * stride_q1h
    DV2 += off_z * stride_q2z + off_h * stride_q2h
    for start_n in range(0, num_block):
        if CAUSAL:
            lo = start_n * BLOCK_M
        else:
            lo = 0
        # initialize row/col offsets
        offs_qm = lo + tl.arange(0, BLOCK_M)
        offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_DMODEL)
        # initialize pointers to value-like data
        q1_ptrs = Q1 + (offs_qm[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        q2_ptrs = Q2 + (offs_qm[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
        k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v1_ptrs = V1 + (offs_n[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        v2_ptrs = V2 + (offs_n[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
        do_ptrs = DO + (offs_qm[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
        dq1_ptrs = DQ1 + (offs_qm[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        dq2_ptrs = DQ2 + (offs_qm[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l_ptrs = L + off_hz * N_CTX
        # initialize dv amd dk
        dv1 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dv2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v1 = tl.load(v1_ptrs)
        v2 = tl.load(v2_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q1, tl.trans(k))
            qk *= qk_scale
            l_i = tl.load(l_ptrs + offs_m_curr)
            p = tl.math.exp2(qk - l_i[:, None])
            # prepare value for backward pass
            v1v2 = v1[:, :, None] * v2[:, None, :]
            
            # compute dv
            do = tl.load(do_ptrs)
            dpv1v2 = do[:, None, :] * q2[:, :, None]
            dv1v2 = tl.sum(p.to(tl.float16)[:, :, None, None] * dpv1v2[:, None, :, :], axis = 0)
            dv1 += tl.sum(dv1v2 * v2[:, None, :], axis = 2)
            dv2 += tl.sum(dv1v2 * v1[:, :, None], axis = 1)

            # compute dq2
            dq2 = tl.load(dq2_ptrs)
            pv1v2 = tl.sum(p.to(tl.float16)[:, :, None, None] * v1v2[None, :, :, :], axis = 1)
            dq2 += tl.sum(do[:, None, :] * pv1v2, axis = 2)
            tl.store(dq2_ptrs, dq2)

            # p -> [TxT]
            # v1 -> [T x d]
            # v2 -> [T x d]
            # q2 -> [T x d]
            # do -> [T x d]
            # v1 x v2 -> [T x d x d]
            # p @ (v1 x v2) -> T x d x d
            # q2 @ p @ (v1 x v2) -> T x d 
            # [d, 1] * [d, d]
            # forward pass code:
            # S_ = v1[:, :, None] * v2[:, None, :]
            # S_ = tl.sum(p.to(tl.float16)[:, :, None, None] *  S_[None, :, :, :], axis = 1)
            # acc *= acc_scale[:, None]
            # acc += tl.sum(q2[:, :, None] * S_, axis = 1)


            # compute dp = dot(v, do)
            Di = tl.load(D_ptrs + offs_m_curr)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp += tl.sum(tl.sum(dpv1v2[:, None, :, :] * v1v2[None, :, :, :], axis = 3), axis = 2)
            # compute ds = p * (dp - delta[:, None])
            ds = p * dp * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds.to(Q1.dtype.element_ty)), q1)
            # compute dq1
            dq1 = tl.load(dq1_ptrs)
            dq1 += tl.dot(ds.to(Q1.dtype.element_ty), k)
            tl.store(dq1_ptrs, dq1)
            # increment pointers
            dq1_ptrs += BLOCK_M * stride_q1m
            dq2_ptrs += BLOCK_M * stride_q2m
            q1_ptrs += BLOCK_M * stride_q1m
            q2_ptrs += BLOCK_M * stride_q2m
            do_ptrs += BLOCK_M * stride_q1m
        # write-back
        dv1_ptrs = DV1 + (offs_n[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        dv2_ptrs = DV2 + (offs_n[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv1_ptrs, dv1)
        tl.store(dv2_ptrs, dv2)
        tl.store(dk_ptrs, dk)


empty = torch.empty(128, device="cuda")


class _ssa(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q1, q2, k, v1, v2, causal, sm_scale):
        # shape constraints
        Lq1, Lq2, Lk, Lv1, Lv2 = q1.shape[-1], q2.shape[-1], k.shape[-1], v1.shape[-1], v2.shape[-1]
        assert Lq1 == Lk and Lq2 == Lv1
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q1)
        BLOCK_M = 16
        BLOCK_N = 16
        grid = (triton.cdiv(q1.shape[2], BLOCK_M), q1.shape[0] * q1.shape[1], 1)
        L = torch.empty((q1.shape[0] * q1.shape[1], q1.shape[2]), device=q1.device, dtype=torch.float32)

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q1, q2, k, v1, v2, sm_scale,
            L,
            o,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v1.stride(0), v1.stride(1), v1.stride(2), v1.stride(3),
            v2.stride(0), v2.stride(1), v2.stride(2), v2.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=4)

        ctx.save_for_backward(q1, q2, k, v1, v2, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 16
        q1, q2, k, v1, v2, o, L = ctx.saved_tensors
        do = do.contiguous()
        dq1 = torch.zeros_like(q1, dtype=torch.float32)
        dq2 = torch.zeros_like(q2, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv1 = torch.empty_like(v1)
        dv2 = torch.empty_like(v2)
        delta = torch.empty_like(L)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q1, q2, k, v1, v2, ctx.sm_scale,
            o, do,
            dq1, dq2, dk, dv1, dv2,
            L, delta,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v1.stride(0), v1.stride(1), v1.stride(2), v1.stride(3),
            v2.stride(0), v2.stride(1), v2.stride(2), v2.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            CAUSAL=ctx.causal,
            num_stages=1,
        )
        return dq1, dq2, dk, dv1, dv2, None, None


ssa = _ssa.apply

# @torch.compile
def ssa_ref(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k: torch.Tensor,
    v1: torch.Tensor,
    v2: torch.Tensor,
    sm_scale: float,
):
  q1, q2, k, v1, v2 = map(lambda x: x.to(torch.float32), [q1, q2, k, v1, v2])
  b, h, l, d_k = q1.shape
  d_v1 = v1.shape[-1]
  d_v2 = v2.shape[-1]
  S = v1.unsqueeze(-1) @ v2.unsqueeze(-2)
  M = torch.tril(torch.ones((l, l), device="cuda"))
  p = torch.matmul(q1, k.transpose(2, 3)) * sm_scale
  p[:, :, M == 0] = float("-inf")
  p = torch.softmax(p.float(), dim=-1).half()
  # S_ = torch.einsum("bhst,bhtd->bhsd", p.float(), S.flatten(start_dim = -2).float()).reshape(b, h, l, d_v1, d_v2)
  # S_ = (p.float()[:, :, :, :, None, None] * S.float()[:, :, None, :, :, :]).sum(3)
  S_ = torch.einsum("bhst,bhtmn->bhsmn", p.float(), S.float())
  o = torch.einsum("bhsm,bhsmn->bhsn", q2, S_)
  # o = (q2[:, :, :, :, None] * S_).sum(-2)
  return o

if __name__ == '__main__':
    B = 4
    H = 2
    L = 1024
    DK = 32
    DV = 32
    require_grad = True
    dtype = torch.float16
    q1 = (torch.rand(B, H, L, DK)).cuda().to(dtype)
    q2 = (torch.rand(B, H, L, DV)).cuda().to(dtype)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).to(dtype)
    v1 = (torch.randn(B, H, L, DV)).cuda().to(dtype)
    v2 = (torch.randn(B, H, L, DV)).cuda().to(dtype)
    q1, q2, k, v1, v2 = map(lambda x: x.requires_grad_(require_grad), [q1, q2, k, v1, v2])

    o  = ssa(q1, q2, k, v1, v2, True, (DK**(-0.5)))
    o2 = ssa_ref(q1, q2, k, v1, v2, (DK**(-0.5)))
    do2 = torch.randn_like(o2)
    o2.backward(do2)
    q1_grad, q1.grad = q1.grad, None
    q2_grad, q2.grad = q2.grad, None  
    k_grad, k.grad = k.grad, None
    v1_grad, v1.grad = v1.grad, None
    v2_grad, v2.grad = v2.grad, None
    o.backward(do2)
    print( (o - o2).abs().max())
    print( (q1.grad - q1_grad).abs().max())
    print( (q2.grad - q2_grad).abs().max())
    print( (k.grad - k_grad).abs().max())
    print( (v1.grad - v1_grad).abs().max())
    print( (v2.grad - v2_grad).abs().max())