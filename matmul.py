import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def is_hip_cdna2():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"

def get_cuda__autotune_config():
    return [
        triton.Config()

    ]