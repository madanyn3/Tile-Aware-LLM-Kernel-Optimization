#!/usr/bin/env python3

import torch, math
import common
from reference import attentionRef
from baseline import attentionSDPA
from typing import List

def makeKVCache (x_batch: int = 1, x_head: int = 16, x_kLen: int = 4096, x_d: int = 128, 
                 x_dtype: torch.dtype = torch.float16, x_device: torch.device = None) -> List[torch.Tensor]:
    l_device = x_device or common.getDevice()
    l_K = torch.randn(x_batch, x_head, x_kLen, x_d, dtype=x_dtype, device=l_device)
    l_V = torch.randn(x_batch, x_head, x_kLen, x_d, dtype=x_dtype, device=x_device)

    return l_K, l_V

def runDecode (x_batch: int = 1, x_head: int = 16, x_kLen: int = 4096, x_d: int = 128, x_blockSteps: int = 128,
               x_warmup: int = 10, x_iters: int = 50, x_dtype: torch.dtype = torch.float16) -> dict:
    l_device = common.getDevice()
    common.setSeed(29)
    l_K16,l_V16 = makeKVCache(x_batch,x_head,x_kLen,x_d,x_dtype=x_dtype,x_device=l_device)
    l_Q16 = torch.randn(x_batch,x_head,1,x_d, dtype=x_dtype, device=l_device)

    # Reference for a single step (correctness check)
    l_Q32, l_K32, l_V32 = l_Q16.float(), l_K16.float(), l_V16.float()
    l_O_ref = attentionRef(l_Q32, l_K32, l_V32, x_causal=True)  # [B,H,1,D]

    # Warmup
    for _ in range(x_warmup):
        _ = attentionSDPA(l_Q16, l_K16, l_V16, x_causal=True)
    common.synchronize()

    # Timed loop: run many 1-token steps against same T to simulate steady state
    times_ms = []
    mem_peaks = []
    for _ in range(x_iters):
        common.resetPeakMemoryStats()
        with common.mScudaTimer() as elapsed_ms:
            for _ in range(x_blockSteps):
                Q16 = torch.randn(x_batch,x_head,1,x_d, dtype=x_dtype, device=l_device)
                _ = attentionSDPA(l_Q16, l_K16, l_V16, x_causal=True)
        t = elapsed_ms()
        times_ms.append(t)
        mem_peaks.append(common.peakMemInMB())

    # Numeric diff (single step; representative)
    O_fast = attentionSDPA(l_Q16, l_K16, l_V16, x_causal=True)
    err_mae = common.mae(O_fast.float() - l_O_ref)
    err_max = common.max_abs(O_fast.float() - l_O_ref)
    err_rms = common.rms(O_fast.float() - l_O_ref)

    avg_ms = sum(times_ms)/len(times_ms)
    total_tokens = x_iters * x_blockSteps * x_batch  # per sequence step; heads are internal
    tokens_per_s = (x_blockSteps * x_batch) / (avg_ms/1000.0)
    report = {
        "B":x_batch, "H":x_head, "T":x_kLen, "D":x_d, "dtype":str(x_dtype).split(".")[-1],
        "avg_ms_per_block": avg_ms,
        "x_blockSteps": x_blockSteps,
        "tokens_per_s": tokens_per_s,
        "mem_peak_MB_avg": sum(mem_peaks)/len(mem_peaks),
        "err_mae": err_mae, "err_rms": err_rms, "err_max": err_max,
    }
    return report

if __name__ == "__main__":
    # l_sweep = [2, 4, 8, 16]
    l_sweep = [1024, 2048, 4096, 8192]
    for T in l_sweep:
        print(runDecode(x_batch=1,x_head=16,x_kLen=T,x_d=128, x_dtype=torch.float16))
