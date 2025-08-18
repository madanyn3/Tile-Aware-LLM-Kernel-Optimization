#!/usr/bin/env python3

"""benchmarking of prefill phase when qLen is the same as kLen and vLen"""

import math, os, torch
from typing import List
import common
from reference import attentionRef
from baseline import attentionSDPA
import warnings, tracemalloc


warnings.filterwarnings("ignore", category=RuntimeWarning)

def makeInputs (x_batch: int = 1, x_head: int = 16, x_qLen: int = 1024, x_d: int = 128, 
                x_dtype: torch.dtype = torch.float16, x_device = None) -> List[torch.Tensor]:
    l_device = x_device or common.getDevice()
    l_Q = torch.randn(x_batch,x_head,x_qLen,x_d, dtype=x_dtype, device=l_device)
    l_K = torch.randn(x_batch,x_head,x_qLen,x_d, dtype=x_dtype, device=l_device)
    l_V = torch.randn(x_batch,x_head,x_qLen,x_d, dtype=x_dtype, device=l_device)
    return l_Q,l_K,l_V

@torch.inference_mode()
def runPrefill(x_batch=1,x_head=16,x_qLen=1024,x_d=128, x_warmup=5, x_iters=20, x_causal=True, x_dtype=torch.float16) -> dict:
    tracemalloc.start()
    l_device = common.getDevice()
    common.setSeed(42)
    l_Q16,l_K16,l_V16 = makeInputs(x_batch,x_head,x_qLen,x_d,x_dtype,l_device)
    # Make reference in fp32
    l_Q32,l_K32,l_V32 = l_Q16.float(), l_K16.float(), l_V16.float()

    # Reference (one pass; correctness only)
    common.synchronize()
    l_O_ref = attentionRef(l_Q32, l_K32, l_V32, x_causal=x_causal)
    common.synchronize()

    # Warmup SDPA
    for _ in range(x_warmup):
        _ = attentionSDPA(l_Q16, l_K16, l_V16, x_causal=x_causal)
    common.synchronize()

    # Timed runs
    times_ms = []
    mem_peaks = []
    for _ in range(x_iters):
        common.resetPeakMemoryStats()
        with common.mScudaTimer() as elapsed_ms:
            O_fast = attentionSDPA(l_Q16, l_K16, l_V16, x_causal=x_causal)
        t = elapsed_ms()
        times_ms.append(t)
        mem_peaks.append(common.peakMemInMB())

    # Correctness (compare promoted O_fast to fp32 ref)
    O_fast32 = O_fast.float()
    err_mae = common.mae(O_fast32 - l_O_ref)
    err_max = common.max_abs(O_fast32 - l_O_ref)
    err_rms = common.rms(O_fast32 - l_O_ref)

    avg_ms = sum(times_ms)/len(times_ms)
    toks_per_s = (x_batch*x_head*x_qLen) / (avg_ms/1000.0)   # per-head-token; also report per-seq if you prefer

    report = {
        "B":x_batch, "H":x_head, "Q_len":x_qLen, "D":x_d, "dtype":str(x_dtype).split(".")[-1],
        "avg_ms_total": avg_ms,
        "ms_per_token": avg_ms/x_qLen,
        "tokens_per_s": toks_per_s,
        "mem_peak_MB_avg": sum(mem_peaks)/len(mem_peaks),
        "err_mae": err_mae, "err_rms": err_rms, "err_max": err_max,
    }
    return report

if __name__ == "__main__":
    l_sweep = [512, 1024, 2048, 4096]
    for QL in l_sweep:
        print(runPrefill( x_batch=1,x_head=16,x_qLen=QL,x_d=128,x_causal=False, x_dtype=torch.float16 ))
