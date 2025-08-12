import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def _fwd_kernel(
    Q1, Q2, K, V, sm_scale,
    L1, L2,
    Out,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
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
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m1_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l1_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m2_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l2_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q1 = tl.load(Q1_block_ptr)
    q1 = (q1 * qk_scale).to(tl.float16)
    q2 = tl.load(Q2_block_ptr)
    q2 = (q2 * qk_scale).to(tl.float16)
    # loop over k, v and update accumulator
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk1 = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk1, float("-inf"))
            qk2 = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk2, float("-inf"))
        qk1 += tl.dot(q1, k)
        qk2 += tl.dot(q2, k)
        # -- compute scaling constant ---
        m1_i_new = tl.maximum(m1_i, tl.max(qk1, 1))
        alpha1 = tl.math.exp2(m1_i - m1_i_new)
        p1 = tl.math.exp2(qk1 - m1_i_new[:, None])

        m2_i_new = tl.maximum(m2_i, tl.max(qk2, 1))
        alpha2 = tl.math.exp2(m2_i - m2_i_new)
        p2 = tl.math.exp2(qk2 - m2_i_new[:, None])
        # -- scale and update acc --
        acc1_scale = l1_i * 0 + alpha1  # workaround some compiler bug
        acc2_scale = l2_i * 0 + alpha2
        acc *= tl.sqrt(acc1_scale * acc2_scale)[:, None]
        acc += tl.dot(tl.sqrt(p1 * p2).to(tl.float16), v)

        # -- update m_i and l_i --
        l1_i = l1_i * alpha1 + tl.sum(p1, 1)
        m1_i = m1_i_new

        l2_i = l2_i * alpha2 + tl.sum(p2, 1)
        m2_i = m2_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / tl.sqrt(l1_i[:, None] * l2_i[:, None])
    l1_ptrs = L1 + off_hz * N_CTX + offs_m
    tl.store(l1_ptrs, m1_i + tl.math.log2(l1_i))
    l2_ptrs = L2 + off_hz * N_CTX + offs_m
    tl.store(l2_ptrs, m2_i + tl.math.log2(l2_i))
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
    Q1, Q2, K, V, sm_scale, Out, DO,
    DQ1, DQ2, DK, DV,
    L1, L2, D,
    stride_q1z, stride_q1h, stride_q1m, stride_q1k,
    stride_q2z, stride_q2h, stride_q2m, stride_q2k,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_v, stride_vk, stride_vn,
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
    V += off_z * stride_q1z + off_h * stride_q1h
    DO += off_z * stride_q1z + off_h * stride_q1h
    DQ1 += off_z * stride_q1z + off_h * stride_q1h
    DQ2 += off_z * stride_q2z + off_h * stride_q2h
    DK += off_z * stride_q1z + off_h * stride_q1h
    DV += off_z * stride_q1z + off_h * stride_q1h
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
        v_ptrs = V + (offs_n[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        do_ptrs = DO + (offs_qm[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        dq1_ptrs = DQ1 + (offs_qm[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        dq2_ptrs = DQ2 + (offs_qm[:, None] * stride_q2m + offs_k[None, :] * stride_q2k)
        # pointer to row-wise quantities in value-like data
        D_ptrs = D + off_hz * N_CTX
        l1_ptrs = L1 + off_hz * N_CTX
        l2_ptrs = L2 + off_hz * N_CTX
        # initialize dv amd dk
        dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
        # k and v stay in SRAM throughout
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
        # loop over rows
        for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
            offs_m_curr = start_m + offs_m
            # load q, k, v, do on-chip
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
            # recompute p = softmax(qk, dim=-1).T
            if CAUSAL:
                qk1 = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
                qk2 = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
            else:
                qk1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                qk2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk1 += tl.dot(q1, tl.trans(k))
            qk2 += tl.dot(q2, tl.trans(k))
            qk1 *= qk_scale
            qk2 *= qk_scale
            l1_i = tl.load(l1_ptrs + offs_m_curr)
            l2_i = tl.load(l2_ptrs + offs_m_curr)
            p1 = tl.math.exp2(qk1 - l1_i[:, None])
            p2 = tl.math.exp2(qk2 - l2_i[:, None])
            p = tl.sqrt(p1 * p2)
            # compute dv
            do = tl.load(do_ptrs)
            dv += tl.dot(tl.trans(p.to(Q1.dtype.element_ty)), do)
            # compute dp = dot(v, do)
            # Di = tl.load(D_ptrs + offs_m_curr)
            # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            dp += tl.dot(do, tl.trans(v))
            # compute ds = p * (dp - delta[:, None])
            ds1 = p * 0.5  * (dp - p1 * dp) * sm_scale
            ds2 = p * 0.5  * (dp - p2 * dp) * sm_scale
            # compute dk = dot(ds.T, q)
            dk += tl.dot(tl.trans(ds1.to(Q1.dtype.element_ty)), q1)
            dk += tl.dot(tl.trans(ds2.to(Q2.dtype.element_ty)), q2)
            # compute dqs
            dq1 = tl.load(dq1_ptrs)
            dq1 += tl.dot(ds1.to(Q1.dtype.element_ty), k)
            tl.store(dq1_ptrs, dq1)
            
            dq2 = tl.load(dq2_ptrs)
            dq2 += tl.dot(ds2.to(Q2.dtype.element_ty), k)
            tl.store(dq2_ptrs, dq2)

            # increment pointers
            dq1_ptrs += BLOCK_M * stride_q1m
            q1_ptrs += BLOCK_M * stride_q1m
            dq2_ptrs += BLOCK_M * stride_q2m
            q2_ptrs += BLOCK_M * stride_q2m
            do_ptrs += BLOCK_M * stride_q1m
        # write-back
        dv_ptrs = DV + (offs_n[:, None] * stride_q1m + offs_k[None, :] * stride_q1k)
        dk_ptrs = DK + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)


empty = torch.empty(128, device="cuda")


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q1, q2, k, v, causal, sm_scale):
        # shape constraints
        Lq, Lk, Lv = q1.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q1)
        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(q1.shape[2], BLOCK_M), q1.shape[0] * q1.shape[1], 1)
        L1 = torch.empty((q1.shape[0] * q1.shape[1], q1.shape[2]), device=q1.device, dtype=torch.float32)
        L2 = torch.empty((q2.shape[0] * q2.shape[1], q2.shape[2]), device=q2.device, dtype=torch.float32)

        num_warps = 4 if Lk <= 64 else 8
        _fwd_kernel[grid](
            q1, q2, k, v, sm_scale,
            L1, L2,
            o,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            num_warps=num_warps,
            num_stages=4)

        ctx.save_for_backward(q1, q2, k, v, o, L1, L2)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        BLOCK = 128
        q1, q2, k, v, o, L1, L2 = ctx.saved_tensors
        do = do.contiguous()
        dq1 = torch.zeros_like(q1, dtype=torch.float32).contiguous()
        dq2 = torch.zeros_like(q2, dtype=torch.float32).contiguous()
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L1)
        _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1],)](
            q1, q2, k, v, ctx.sm_scale,
            o, do,
            dq1, dq2, dk, dv,
            L1, L2, delta,
            q1.stride(0), q1.stride(1), q1.stride(2), q1.stride(3),
            q2.stride(0), q2.stride(1), q2.stride(2), q2.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q1.shape[0], q1.shape[1], q1.shape[2],
            ctx.grid[0],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL, num_warps=8,
            CAUSAL=ctx.causal,
            num_stages=1,
        )
        return dq1, dq2, dk, dv, None, None


attention_sq = _attention.apply

@torch.compile
def attention_square(
    q1: torch.Tensor,
    q2: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
):
  q1, q2, k, v = map(lambda x: x.to(torch.float32), [q1, q2, k, v])
  b, h, l, d_k = q1.shape
  d_v = v.shape[-1]
  M = torch.tril(torch.ones((l, l), device="cuda"))
  p1 = torch.matmul(q1, k.transpose(2, 3)) * sm_scale
  p1[:, :, M == 0] = float("-inf")
  p1 = torch.softmax(p1.float() - p1.float().amax(-1, keepdim = True), dim=-1)

  p2 = torch.matmul(q2, k.transpose(2, 3)) * sm_scale
  p2[:, :, M == 0] = float("-inf")
  p2 = torch.softmax(p2.float() - p2.float().amax(-1, keepdim = True), dim=-1)
  o = torch.matmul(torch.sqrt(p1 * p2 + 1e-8), v)
  return o

if __name__ == '__main__':
    B = 4
    H = 2
    L = 1024
    DK = 32
    DV = 32
    require_grad = True
    dtype = torch.float16
    q1 = (torch.randn(B, H, L, DK)).cuda().to(dtype)
    q2 = (torch.randn(B, H, L, DK)).cuda().to(dtype)
    k = (torch.randn(B, H, L, DK)).cuda()
    k = torch.nn.functional.normalize(k, dim=-1, p=2).to(dtype)
    v = (torch.randn(B, H, L, DV)).cuda().to(dtype)
    q1, q2, k, v = map(lambda x: x.requires_grad_(require_grad), [q1, q2, k, v])

    o  = attention_sq(q1, q2, k, v, True, (DK**(-0.5)))
    o2 = attention_square(q1, q2, k, v, (DK**(-0.5)))
    do2 = torch.randn_like(o2)
    o2.backward(do2)
    q1_grad, q1.grad = q1.grad, None
    q2_grad, q2.grad = q2.grad, None  
    k_grad, k.grad = k.grad, None
    v_grad, v.grad = v.grad, None
    o.backward(do2)
    print( (o - o2).abs().max())
    print( (q1.grad - q1_grad).abs().max())
    print( (q2.grad - q2_grad).abs().max())
    print( (k.grad - k_grad).abs().max())
    print( (v.grad - v_grad).abs().max())