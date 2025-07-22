import torch
import triton
import triton.language as tl

@triton.jit
def outer_product_kernel(
    u_ptr,  # Pointer to the first input vector (M elements)
    v_ptr,  # Pointer to the second input vector (N elements)
    output_ptr,  # Pointer to the output matrix (M x N elements)
    M: tl.constexpr,  # Dimension of vector u
    N: tl.constexpr,  # Dimension of vector v
    BLOCK_SIZE_M: tl.constexpr, # Number of elements in M dimension each program should process
    BLOCK_SIZE_N: tl.constexpr, # Number of elements in N dimension each program should process
):
    pid_m = tl.program_id(axis=0)  # Program ID for M dimension
    pid_n = tl.program_id(axis=1)  # Program ID for N dimension

    # Compute offsets for loading u
    u_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    u_mask = u_offsets < M

    # Compute offsets for loading v
    v_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    v_mask = v_offsets < N

    # Load u and v elements
    u = tl.load(u_ptr + u_offsets, mask=u_mask, other=0.0)
    v = tl.load(v_ptr + v_offsets, mask=v_mask, other=0.0)

    # Calculate outer product for the block
    output_block = u[:, None] * v[None, :] 

    # Compute offsets for storing output
    output_offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    output_offsets_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Store results in the output matrix
    tl.store(output_ptr + output_offsets_m[:, None] * N + output_offsets_n[None, :], output_block, 
            mask=u_mask[:, None] & v_mask[None, :])

M, N = 64, 128
u = torch.randn((M, ), dtype = torch.bfloat16).cuda()
v = torch.randn((N, ), dtype = torch.bfloat16).cuda()
o = torch.zeros((M, N), dtype = torch.bfloat16).cuda()
BLOCK_SIZE_M = 4
BLOCK_SIZE_N = 4
grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
outer_product_kernel[grid](u, v, o, M, N, BLOCK_SIZE_M, BLOCK_SIZE_N)
print(o - u[:, None] @ v[None, :])