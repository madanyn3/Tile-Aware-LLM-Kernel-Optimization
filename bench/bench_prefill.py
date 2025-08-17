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
def run_prefill(B=1,H=16,QL=1024,D=128, warmup=5, iters=20, causal=True, dtype=torch.float16):
    tracemalloc.start()
    l_device = common.getDevice()
    common.setSeed(42)
    Q16,K16,V16 = makeInputs(B,H,QL,D,dtype,l_device)
    # Make reference in fp32
    Q32,K32,V32 = Q16.float(), K16.float(), V16.float()

    # Reference (one pass; correctness only)
    common.synchronize()
    O_ref = attentionRef(Q32, K32, V32, x_causal=causal)
    common.synchronize()

    # Warmup SDPA
    for _ in range(warmup):
        _ = attentionSDPA(Q16, K16, V16, x_causal=causal)
    common.synchronize()

    # Timed runs
    times_ms = []
    mem_peaks = []
    for _ in range(iters):
        common.resetPeakMemoryStats()
        with common.mScudaTimer() as elapsed_ms:
            O_fast = attentionSDPA(Q16, K16, V16, x_causal=causal)
        t = elapsed_ms()
        times_ms.append(t)
        mem_peaks.append(common.peakMemInMB())

    # Correctness (compare promoted O_fast to fp32 ref)
    O_fast32 = O_fast.float()
    err_mae = common.mae(O_fast32 - O_ref)
    err_max = common.max_abs(O_fast32 - O_ref)
    err_rms = common.rms(O_fast32 - O_ref)

    avg_ms = sum(times_ms)/len(times_ms)
    toks_per_s = (B*H*QL) / (avg_ms/1000.0)   # per-head-token; also report per-seq if you prefer

    report = {
        "B":B, "H":H, "Q_len":QL, "D":D, "dtype":str(dtype).split(".")[-1],
        "avg_ms_total": avg_ms,
        "ms_per_token": avg_ms/QL,
        "tokens_per_s": toks_per_s,
        "mem_peak_MB_avg": sum(mem_peaks)/len(mem_peaks),
        "err_mae": err_mae, "err_rms": err_rms, "err_max": err_max,
    }
    return report

if __name__ == "__main__":
    for QL in [512, 1024, 2048, 4096]:
        print(run_prefill( B=1,H=16,QL=QL,D=128, dtype=torch.float16 ))
