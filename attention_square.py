import torch
# @torch.compile
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
  S = v.unsqueeze(-1) @ v2.unsqueeze(-2)
  M = torch.tril(torch.ones((l, l), device="cuda"))
  p1 = torch.matmul(q1, k.transpose(2, 3)) * sm_scale
  p1[:, :, M == 0] = float("-inf")
  p1 = torch.softmax(p1.float(), dim=-1).half()

  p2 = torch.matmul(q2, k.transpose(2, 3)) * sm_scale
  p2[:, :, M == 0] = float("-inf")
  p2 = torch.softmax(p1.float(), dim=-1).half()
  o = torch.matmul((p1 * p2), v)
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
    v = (torch.randn(B, H, L, DV)).cuda().to(dtype)
    q1, q2, k, v = map(lambda x: x.requires_grad_(require_grad), [q1, q2, k, v])

    # o  = ssa(q1, q2, k, v1, v2, True, (DK**(-0.5)))
    o2 = ssa_ref(q1, q2, k, v1, v2, (DK**(-0.5)))
    # do2 = torch.randn_like(o2)
    o2.backward(do2)
    # q1_grad, q1.grad = q1.grad, None
    # q2_grad, q2.grad = q2.grad, None  
    # k_grad, k.grad = k.grad, None
    # v1_grad, v1.grad = v1.grad, None
    # v2_grad, v2.grad = v2.grad, None
    # o.backward(do2)
    # print( (o - o2).abs().max())
    # print( (q1.grad - q1_grad).abs().max())
    # print( (q2.grad - q2_grad).abs().max())
    # print( (k.grad - k_grad).abs().max())
    # print( (v1.grad - v1_grad).abs().max())
    # print( (v2.grad - v2_grad).abs().max())