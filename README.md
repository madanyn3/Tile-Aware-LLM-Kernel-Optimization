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

---

## Next Steps

### Phase 1 – Naïve Attention Kernel
- Compute `S = Q @ Kᵀ`.
- Apply scale factor `1/√d`.
- Implement row-wise softmax (multi-pass, unoptimized).
- Compute `O = softmax(S) @ V`.
- Validate against PyTorch reference for small sizes (e.g., 128×128×64).
- Goal: correctness first, performance later.

---

## Roadmap
- Phase 2: Minimal tiled attention kernel (fused softmax, single-tile).
- Phase 3: Streaming softmax for long sequences.
- Phase 4: Quantization-aware kernels.
- Phase 5: Autotuning and benchmarks.
- Phase 6: Documentation and stretch goals.

---

## Repo Structure
.
├── README.md
├── bench
│   ├── __pycache__
│   │   ├── baseline.cpython-38.pyc
│   │   ├── common.cpython-38.pyc
│   │   └── reference.cpython-38.pyc
│   ├── opencl
│   │   ├── __pycache__
│   │   │   └── common.cpython-312.pyc
│   │   ├── common.py
│   │   └── tiled_matmul.py
│   └── pytorch
│       ├── baseline.py
│       ├── bench_decode.py
│       ├── bench_prefill.py
│       ├── common.py
│       ├── reference.py
│       └── sweep.py
└── kernels
    ├── matmul_tiled.cl
    ├── vec_add.cl
    └── vec_add.py