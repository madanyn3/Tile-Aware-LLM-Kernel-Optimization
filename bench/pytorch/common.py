#!/usr/bin/env python3

import math, os, time, sys
import torch
import warnings, tracemalloc
import contextlib

def getDevice ():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        warnings.warn("cuda not available", RuntimeWarning)
        return torch.device("cpu")
    
def synchronize ():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        pass

def resetPeakMemoryStats ():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        if sys.version_info >= (3, 9):
            tracemalloc.reset_peak()
        else:
            tracemalloc.stop()
            tracemalloc.start()
    
    
def setSeed (x_seed=224):
    torch.manual_seed(x_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(x_seed)

def peakMemInMB ():
    if torch.cuda.is_available():
        l_mem = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        return l_mem
    else:
        _, l_mem = tracemalloc.get_traced_memory()
        resetPeakMemoryStats()
        return l_mem / 1024**2

@contextlib.contextmanager
def mScudaTimer ():
    if not torch.cuda.is_available():
        l_startTime = time.perf_counter_ns()
        yield lambda: (time.perf_counter_ns() - l_startTime) / 1000000.0
        return
    l_s = torch.cuda.Stream()
    l_start, l_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    with torch.cuda.stream(l_s):
        l_start.record()
        yield lambda: (l_end.elapsed_time(l_start)) 
        l_end.record()
    torch.cuda.synchronize()

def rms (l_a: torch.Tensor) -> float: return (l_a.square().mean().sqrt()).item()

def mae (l_a: torch.Tensor) -> float: return (l_a.abs().mean()).item()

def max_abs (l_a: torch.Tensor) -> float: return (l_a.abs().max()).item()