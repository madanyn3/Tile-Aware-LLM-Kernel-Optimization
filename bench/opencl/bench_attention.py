# bench/naive_attention.py
# Naive immpementation of attention using small openCL kernels

import numpy as np
import pyopencl as cl
import torch
import time, os
from common import pickOpenCLDevice, buildOpenCLProgramFromPath, buildOpenClProgramFromString

KERNEL_DIR = os.path.join(os.path.dirname(__file__), "..\\..", "kernels")

def runNaiveAttention (ctx, queue, program, M, N, D, type, TILE):
    Q = np.random.randn(M, D).astype(type)
    K = np.random.randn(N, D).astype(type)
    V = np.random.randn(N, D).astype(type)
    Kdim = D

    matmul_tiled = program.matmul_tiled
    row_max = program.row_max
    row_norm = program.row_norm
    row_exp_sum = program.row_exp_sum
    scale_inplace = program.scale_inplace

    mf = cl.mem_flags

    Q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
    K_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
    V_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
    S_buf = cl.Buffer(ctx, mf.READ_WRITE, M * N * np.dtype(type).itemsize)   
    O_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * D * np.dtype(type).itemsize)   

    row_max_buf = cl.Buffer(ctx, mf.READ_WRITE, M * np.dtype(type).itemsize)
    row_sum_buf = cl.Buffer(ctx, mf.READ_WRITE, M * np.dtype(type).itemsize)

    def roundUp (x, tile): return ((x + tile - 1) // tile) * tile

    def evt_ms(evt):
        evt.wait()
        return (evt.profile.end - evt.profile.start) * 1e-6
    
    global_qk = (roundUp(N, TILE,), roundUp(M, TILE))
    global_pv = (roundUp(D, TILE), roundUp(M, TILE))
    local_2d  = (TILE, TILE)

    # --- pipeline ---
    t0 = time.perf_counter()

    # -----------------------------
    # Step 1: S = Q @ K^T
    # -----------------------------
    evt_qk = matmul_tiled(
        queue, global_qk, local_2d,
        np.int32(M), np.int32(N), np.int32(Kdim),
        Q_buf, np.int32(Kdim),
        K_buf, np.int32(Kdim),
        S_buf, np.int32(N)
    )

    # -----------------------------
    # Step 2: S = S/sqrt(D)
    # -----------------------------
    scale = np.float32(1.0 / np.sqrt(D))
    evt_scale = scale_inplace(
        queue, (M,), None,
        np.int32(M), np.int32(N), scale, S_buf,
        wait_for=[evt_qk]
    )

    # -----------------------------
    # Step 3: row_max = max(S, axis=1) 
    # -----------------------------
    evt_max = row_max(
        queue, (M,), None,
        np.int32(M), np.int32(N),
        S_buf, row_max_buf,
        wait_for=[evt_scale]
    )

    # -----------------------------
    # Step 4: S = exp(S - row_max)
    # -----------------------------
    evt_exp_sum = row_exp_sum(
        queue, (M,), None,
        np.int32(M), np.int32(N),
        S_buf, row_max_buf, row_sum_buf,
        wait_for=[evt_max]
    )

    # -----------------------------
    # Step 5: S = S / row_sum
    # -----------------------------
    evt_norm = row_norm(
        queue, (M,), None,
        np.int32(M), np.int32(N),
        S_buf, row_sum_buf,
        wait_for=[evt_exp_sum]
    )

    # -----------------------------
    # Step 6: O = S @ V
    # -----------------------------
    evt_pv = matmul_tiled(
        queue, global_pv, local_2d,
        np.int32(M), np.int32(D), np.int32(N),
        S_buf, np.int32(N),
        V_buf, np.int32(N),
        O_buf, np.int32(D),
        wait_for=[evt_norm]
    )

    evt_pv.wait()
    queue.finish()
    t1 = time.perf_counter()

    O = np.empty((M, D), dtype=type)
    cl.enqueue_copy(queue, O, O_buf).wait()

    # --- reference using torch ---
    Q_t = torch.from_numpy(Q)
    K_t = torch.from_numpy(K)
    V_t = torch.from_numpy(V)
    start = time.perf_counter()
    S_ref = (Q_t @ K_t.t()) / np.sqrt(D)
    S_ref = torch.softmax(S_ref, dim=1)
    O_ref = (S_ref @ V_t).numpy()
    end = time.perf_counter()

    avg_cpu_ms = (end - start) * 1e3
    err = np.max(np.abs(O_ref - O))
    mse = np.mean((O_ref - O)**2)
    # print(f"attn max-abs err: {err:.3e}, mse: {mse:.3e}")
    avg_total_ms = (t1 - t0) * 1e3
    avg_kernel_ms = (evt_ms(evt_qk) + evt_ms(evt_scale) + evt_ms(evt_max) +
                     evt_ms(evt_exp_sum) + evt_ms(evt_norm) + evt_ms(evt_pv))
    
    return avg_cpu_ms, avg_total_ms, avg_kernel_ms, err, mse

def runAttentionFusedSoftmax (ctx, queue, program, M, N, D, type, TILE):
    """This implemets attention there kernels i.e. QK^T, Softmax, SV"""

    Q = np.random.randn(M, D).astype(type)
    K = np.random.randn(N, D).astype(type)
    V = np.random.randn(N, D).astype(type)
    Kdim = D

    matmul_tiled = program.matmul_tiled
    softmax = program.softmax
    softmax2 = program.softmax2 # softmax2 substracts row_max before taking exp

    mf = cl.mem_flags

    Q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
    K_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
    V_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
    S_buf = cl.Buffer(ctx, mf.READ_WRITE, M * N * np.dtype(type).itemsize)   
    O_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * D * np.dtype(type).itemsize)

    def roundUp (x, tile): return ((x + tile - 1) // tile) * tile

    def evt_ms(evt):
        evt.wait()
        return (evt.profile.end - evt.profile.start) * 1e-6
    
    global_qk = (roundUp(N, TILE,), roundUp(M, TILE))
    global_pv = (roundUp(D, TILE), roundUp(M, TILE))
    local_2d  = (TILE, TILE)
    local_size = (N,)
    shared_mem = N * np.dtype(type).itemsize

    t0 = time.perf_counter()

    # -----------------------------
    # Step 1: S = Q @ K^T
    # -----------------------------
    evt_qk = matmul_tiled(
        queue, global_qk, local_2d,
        np.int32(M), np.int32(N), np.int32(Kdim),
        Q_buf, np.int32(Kdim),
        K_buf, np.int32(Kdim),
        S_buf, np.int32(N)
    )
    _, device = pickOpenCLDevice()
    if N > device.max_work_group_size or N > device.max_work_item_sizes[0]:
        raise RuntimeError("Chosen local size exceeds device limits. Recompute L or use multi-group softmax.")


    # -----------------------------
    # Step 2: S = softmax(S * scale)
    # -----------------------------
    scale = np.float32(1.0 / np.sqrt(D))
    evt_softmax = softmax(
        queue, (M*N,), local_size,
        np.int32(M), np.int32(N), scale,
        S_buf, cl.LocalMemory(shared_mem),
        wait_for=[evt_qk]
    )

    # -----------------------------
    # Step 3: O = S @ V
    # -----------------------------
    evt_pv = matmul_tiled(
        queue, global_pv, local_2d,
        np.int32(M), np.int32(D), np.int32(N),
        S_buf, np.int32(N),
        V_buf, np.int32(N),
        O_buf, np.int32(D),
        wait_for=[evt_softmax]
    )

    evt_pv.wait()
    queue.finish()
    t1 = time.perf_counter()

    O = np.empty((M, D), dtype=type)
    cl.enqueue_copy(queue, O, O_buf).wait()

    # --- reference using torch ---
    Q_t = torch.from_numpy(Q)
    K_t = torch.from_numpy(K)
    V_t = torch.from_numpy(V)
    start = time.perf_counter()
    S_ref = (Q_t @ K_t.t()) / np.sqrt(D)
    S_ref = torch.softmax(S_ref, dim=1)
    O_ref = (S_ref @ V_t).numpy()
    end = time.perf_counter()

    avg_cpu_ms = (end - start) * 1e3
    err = np.max(np.abs(O_ref - O))
    mse = np.mean((O_ref - O)**2)
    # print(f"attn max-abs err: {err:.3e}, mse: {mse:.3e}")
    avg_total_ms = (t1 - t0) * 1e3
    avg_kernel_ms = (evt_ms(evt_qk) + evt_ms(evt_softmax) + evt_ms(evt_pv))
    
    return avg_cpu_ms, avg_total_ms, avg_kernel_ms, err, mse

def runAttentionFusedSoftmax2 (ctx, queue, program, M, N, D, type, TILE):
    """This implemets attention there kernels i.e. QK^T, Softmax, SV"""

    Q = np.random.randn(M, D).astype(type)
    K = np.random.randn(N, D).astype(type)
    V = np.random.randn(N, D).astype(type)
    Kdim = D

    matmul_tiled = program.matmul_tiled
    softmax = program.softmax
    softmax2 = program.softmax2 # softmax2 substracts row_max before taking exp

    mf = cl.mem_flags

    Q_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=Q)
    K_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=K)
    V_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=V)
    S_buf = cl.Buffer(ctx, mf.READ_WRITE, M * N * np.dtype(type).itemsize)   
    O_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M * D * np.dtype(type).itemsize)

    def roundUp (x, tile): return ((x + tile - 1) // tile) * tile

    def evt_ms(evt):
        evt.wait()
        return (evt.profile.end - evt.profile.start) * 1e-6
    
    global_qk = (roundUp(N, TILE,), roundUp(M, TILE))
    global_pv = (roundUp(D, TILE), roundUp(M, TILE))
    local_2d  = (TILE, TILE)

    t0 = time.perf_counter()
    local_size = (N,)
    shared_mem = N * np.dtype(type).itemsize

    # -----------------------------
    # Step 1: S = Q @ K^T
    # -----------------------------
    evt_qk = matmul_tiled(
        queue, global_qk, local_2d,
        np.int32(M), np.int32(N), np.int32(Kdim),
        Q_buf, np.int32(Kdim),
        K_buf, np.int32(Kdim),
        S_buf, np.int32(N)
    )

    # -----------------------------
    # Step 2: S = softmax(S * scale)
    # -----------------------------
    scale = np.float32(1.0 / np.sqrt(D))
    evt_softmax = softmax2(
        queue, (M*N,), local_size,
        np.int32(M), np.int32(N), scale,
        S_buf, cl.LocalMemory(shared_mem),
        wait_for=[evt_qk]
    )

    # -----------------------------
    # Step 3: O = S @ V
    # -----------------------------
    evt_pv = matmul_tiled(
        queue, global_pv, local_2d,
        np.int32(M), np.int32(D), np.int32(N),
        S_buf, np.int32(N),
        V_buf, np.int32(N),
        O_buf, np.int32(D),
        wait_for=[evt_softmax]
    )

    evt_pv.wait()
    queue.finish()
    t1 = time.perf_counter()

    O = np.empty((M, D), dtype=type)
    cl.enqueue_copy(queue, O, O_buf).wait()

    # --- reference using torch ---
    Q_t = torch.from_numpy(Q)
    K_t = torch.from_numpy(K)
    V_t = torch.from_numpy(V)
    start = time.perf_counter()
    S_ref = (Q_t @ K_t.t()) / np.sqrt(D)
    S_ref = torch.softmax(S_ref, dim=1)
    O_ref = (S_ref @ V_t).numpy()
    end = time.perf_counter()

    avg_cpu_ms = (end - start) * 1e3
    err = np.max(np.abs(O_ref - O))
    mse = np.mean((O_ref - O)**2)
    # print(f"attn max-abs err: {err:.3e}, mse: {mse:.3e}")
    avg_total_ms = (t1 - t0) * 1e3
    avg_kernel_ms = (evt_ms(evt_qk) + evt_ms(evt_softmax) + evt_ms(evt_pv))
    
    return avg_cpu_ms, avg_total_ms, avg_kernel_ms, err, mse




def main():
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from common import pickOpenCLDevice, buildOpenCLProgramFromPath, buildOpenClProgramFromString

    platform, device = pickOpenCLDevice()
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx, device=device,
                            properties=cl.command_queue_properties.PROFILING_ENABLE)

    srcNaiveAttn = ''
    with open(os.path.join(KERNEL_DIR, "matmul_tiled.cl"), 'r') as f:
        srcNaiveAttn += f.read() + '\n'
    with open(os.path.join(KERNEL_DIR, "row_max.cl"), 'r') as f:
        srcNaiveAttn += f.read() + '\n'
    with open(os.path.join(KERNEL_DIR, "row_norm.cl"), 'r') as f:
        srcNaiveAttn += f.read() + '\n'
    with open(os.path.join(KERNEL_DIR, "row_exp_sum.cl"), 'r') as f:
        srcNaiveAttn += f.read() + '\n'
    with open(os.path.join(KERNEL_DIR, "scale_inplace.cl"), 'r') as f:
        srcNaiveAttn += f.read() + '\n'
    program = buildOpenClProgramFromString(ctx, srcNaiveAttn)

    srcFusedSoftmaxAttn = ''
    with open(os.path.join(KERNEL_DIR, "matmul_tiled.cl"), 'r') as f:
        srcFusedSoftmaxAttn += f.read() + '\n'
    with open(os.path.join(KERNEL_DIR, "softmax.cl"), 'r') as f:
        srcFusedSoftmaxAttn += f.read() + '\n'

    M, N, D = 256, 256, 64
    TILE = 16
    type = np.float32

    print("==== Naive Attention ====")
    avg_cpu_ms, avg_total_ms, avg_kernel_ms, err, mse = runNaiveAttention(ctx, queue, program, M, N, D, type, TILE)
    print(f"Naive Attention M={M}, N={N}, D={D}, tile={TILE}, type={type.__name__}")
    print(f"  CPU ref time (ms): {avg_cpu_ms:.3f}")
    print(f"  Total GPU time (ms): {avg_total_ms:.3f}")
    print(f"  Kernel GPU time (ms): {avg_kernel_ms:.3f}")
    print(f"  max-abs err: {err:.3e}, mse: {mse:.3e}")

    print("==== Fused Softmax Attention ====")
    program2 = buildOpenClProgramFromString(ctx, srcFusedSoftmaxAttn)
    avg_cpu_ms, avg_total_ms, avg_kernel_ms, err, mse = runAttentionFusedSoftmax(ctx, queue, program2, M, N, D, type, TILE)
    print(f"Fused Softmax Attention M={M}, N={N}, D={D}, tile={TILE}, type={type.__name__}")
    print(f"  CPU ref time (ms): {avg_cpu_ms:.3f}")
    print(f"  Total GPU time (ms): {avg_total_ms:.3f}")
    print(f"  Kernel GPU time (ms): {avg_kernel_ms:.3f}")
    print(f"  max-abs err: {err:.3e}, mse: {mse:.3e}")

    print("==== Fused Softmax2 Attention ====")
    program3 = buildOpenClProgramFromString(ctx, srcFusedSoftmaxAttn)
    avg_cpu_ms, avg_total_ms, avg_kernel_ms, err, mse = runAttentionFusedSoftmax2(ctx, queue, program3, M, N, D, type, TILE)
    print(f"Fused Softmax2 Attention M={M}, N={N}, D={D}, tile={TILE}, type={type.__name__}")
    print(f"  CPU ref time (ms): {avg_cpu_ms:.3f}")
    print(f"  Total GPU time (ms): {avg_total_ms:.3f}")
    print(f"  Kernel GPU time (ms): {avg_kernel_ms:.3f}")
    print(f"  max-abs err: {err:.3e}, mse: {mse:.3e}")

if __name__ == "__main__":
    main()