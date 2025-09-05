# bench/opencl/tiled_matmul.py
# Benchmark tiled matrix multiplication using OpenCL

import os, time
import numpy as np
import pyopencl as cl
import torch
from common import pickOpenCLDevice, buildOpenCLProgram

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..\\..", "kernels")

def runTiledMatmul(ctx, queue, program, M, K, N, tile, type):
    A = np.random.randn(M, K).astype(type)
    B = np.random.randn(K, N).astype(type)
    C = np.empty((M, N), dtype=type)

    mf = cl.mem_flags
    A_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=A)
    B_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
    C_buf = cl.Buffer(ctx, mf.WRITE_ONLY, C.nbytes)

    kernel = program.matmul_tiled

    kernel.set_args(np.int32(M), np.int32(N), np.int32(K),
                    A_buf, np.int32(K), 
                    B_buf, np.int32(N), 
                    C_buf, np.int32(N))

    TILE = tile
    global_size = ( (N + TILE - 1) // TILE * TILE, (M + TILE - 1) // TILE * TILE )
    local_size = (TILE, TILE)
    evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
    evt.wait()
    cl.enqueue_copy(queue, C, C_buf).wait()

    # reference using torch
    A_t = torch.from_numpy(A)
    B_t = torch.from_numpy(B)
    start = time.perf_counter()
    C_ref = (A_t @ B_t).numpy()
    end = time.perf_counter()
    # print(f"CPU matmul time: {(end - start)*1e3:.3f} ms")
    avg_cpu_ms = (end - start) * 1e3
    err = np.max(np.abs(C_ref - C))
    mse = np.mean((C_ref - C)**2)
    # print(f"matmul max-abs err: {err:.3e}, mse: {mse:.3e}")

    # timing
    t0 = time.perf_counter()
    for _ in range(10):
        evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        evt.wait()
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) / 10 * 1000
    # print(f"matmul avg kernel time (ms): {avg_ms:.3f}")
    return avg_cpu_ms, avg_ms

def main():
    platform, device = pickOpenCLDevice()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, device=device)

    mm_kernel = buildOpenCLProgram(ctx, os.path.join(KERNEL_DIR, "matmul_tiled.cl"))

    # M, K, N = 1024, 2048, 1024
    matrix_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096]
    tile = 16
    for size in matrix_sizes:
        M = N = size
        K = 2 * size
        print(f"Matrix size: {M}x{K} * {K}x{N}, tile={tile}")
        cpu_time = []
        gpu_time = []
        for itr in range(3):
            cpu_ms, gpu_ms = runTiledMatmul(ctx, queue, mm_kernel, M, K, N, tile)
            cpu_time.append(cpu_ms)
            gpu_time.append(gpu_ms)
        print(f'M:{M} K:{K} N:{N} | CPU avg time: {np.mean(cpu_time):.3f} ms | GPU avg time: {np.mean(gpu_time):.3f} ms')

if __name__ == "__main__":
    main()