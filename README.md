# Tile-Aware-LLM-Kernel-Optimization

An experimental project to build transformer attention kernels from scratch in OpenCL, starting with basic matrix multiplication and moving towards optimized, tile-aware kernels.

---

## Progress

### Phase 0 – Setup and Tiled Matmul
- **PyTorch baselines (`bench/pytorch/`)**
  - `baseline.py`, `reference.py`: Ground-truth attention and matmul implementations.
  - `bench_prefill.py`, `bench_decode.py`: Prefill and decode benchmarks for small transformer workloads.
  - `sweep.py`: Simple parameter sweeps for testing correctness and runtime.
  - Validated PyTorch reference vs baseline for both prefill and decode paths.

- **OpenCL setup (`bench/opencl/`)**
  - `tiled_matmul.py`: Host-side harness to run tiled GEMM in OpenCL.
  - `common.py`: Shared utilities for buffer allocation and execution.
  - Verified OpenCL outputs against PyTorch with small and large problem sizes.

- **Kernel implementation (`kernels/matmul_tiled.cl`)**
  - Shared-memory tiled GEMM kernel (16×16 tile per work-group).
  - Correctness validated against PyTorch/NumPy reference.
  - Example benchmark (`M=1024, N=2048, K=1024`):  
    - CPU (NumPy): ~23.5 ms  
    - OpenCL kernel: ~12.3 ms  
    - Accuracy: max-abs error ≈ 2e-4, MSE ≈ 3.7e-10

### Phase 1 – Naïve Attention Kernel
- Implemented the first end-to-end attention pipeline in OpenCL:
  - Compute `S = Q @ Kᵀ` using tiled GEMM kernel.
  - Apply scale factor `1/√d`.
  - Row-wise softmax:
    - Started with multi-pass kernels (row max, exponentiation, row sum, normalize).
    - Then fused into a single softmax kernel for efficiency.
  - Compute `O = softmax(S) @ V` using tiled GEMM.
- Validation against PyTorch reference:
  - For small cases (e.g., `M=N=128, D=64`), outputs match within error tolerance.
  - Example benchmark (`M=1024, N=1024, D=64`):  
    - CPU reference: ~26 ms  
    - OpenCL total time: ~45 ms  
    - OpenCL kernel time: ~8.4 ms  
    - Accuracy: max-abs error ≈ 0.5, MSE ≈ 2.6e-3
- Fused softmax further reduced kernel time (≈0.18 ms), but current implementation is limited by work-group size (≤256). Next step is to introduce **row tiling** for scalability.

---

## Roadmap
- Phase 2: Minimal tiled attention kernel (fused softmax, single-tile).
- Phase 3: Streaming softmax for long sequences.
- Phase 4: Quantization-aware kernels.
- Phase 5: Autotuning and benchmarks.
- Phase 6: Documentation and stretch goals.

---

## Repo Structure
.<br>
├── README.md<br>
├── bench<br>
│   ├── opencl<br>
|   |   ├── bench_attention.py<br>
│   │   ├── common.py<br>
│   │   └── tiled_matmul.py<br>
│   └── pytorch<br>
│       ├── baseline.py<br>
│       ├── bench_decode.py<br>
│       ├── bench_prefill.py<br>
│       ├── common.py<br>
│       ├── reference.py<br>
│       └── sweep.py<br>
└── kernels<br>
    ├── attention_naive.cl<br>
    ├── matmul_tiled.cl<br>
    ├── row_exp_sum.cl<br>
    ├── row_max.cl<br>
    ├── row_norm.cl<br>
    ├── scale_inplace.cl<br>
    ├── softmax.cl<br>
    ├── vec_add.cl<br>
    └── vec_add.py<br>
