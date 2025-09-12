// kernels/softmax.cl
// Compute the softmax of each row of a 2D matrix S (size MxN)
// and store the results back in S.

__kernel void softmax (
    const int M, const int N,
    const float scale,
    __global float* S,
    __local float* exp_values)
{
    int row = get_global_id(0);
    int lid = get_local_id(0);
    if (row >= M) return;

    float val = S[row * N + lid] * scale;
    exp_values[lid] = exp(val);
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
    for (int i = 0; i < N; i++)
        sum += exp_values[i];

    S[row * N + lid] = exp_values[lid] / sum;
    return;
}

__kernel void softmax2 (
    const int M, const int N,
    const float scale,
    __global float* S,
    __local float* shared)  
{
    int gid = get_global_id(0);      // global thread id (0..M*N-1)
    int row = get_group_id(0);       // which row this group handles
    int lid = get_local_id(0);       // column index within row
    int L = get_local_size(0);

    if (row >= M || lid >= N) return;

    float val = S[row * N + lid] * scale;
    shared[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // max reduce (guarded)
    for (int stride = L >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float a = shared[lid];
            float b = (lid + stride < N) ? shared[lid + stride] : -INFINITY;
            shared[lid] = fmax(a, b);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_max = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float e = exp(val - row_max);
    shared[lid] = e;
    barrier(CLK_LOCAL_MEM_FENCE);

    // sum reduce (guarded)
    for (int stride = L >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            float a = shared[lid];
            float b = (lid + stride < N) ? shared[lid + stride] : 0.0f;
            shared[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float row_sum = shared[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    S[row * N + lid] = e / row_sum;
}


// Fused row-tiled streaming softmax (single kernel).
// - One work-group per row
// - Work-group size = L (threads per tile), chosen on host <= device limits
// - Kernel processes the row in tiles of L columns: [0..L-1], [L..2L-1], ...
// - First loop: compute row_max (m) and running sum s using numerically-stable streaming update
// - Second loop: write normalized outputs exp(x - m) / s in-place
//
// Signature:
// __kernel void softmax_fused(
//     const int M, const int N, const float scale,
//     __global float* S,      // input: S (MxN), overwritten with softmax outputs
//     __local float* lbuf )   // local buffer, length >= L
//
#pragma OPENCL EXTENSION cl_khr_fp64 : enable   // optional; kernel uses float32

__kernel void softmax_tiledFused(
    const int M,
    const int N,
    const float scale,
    __global float* S,
    __local float* lbuf)   
{
    const int gid = get_global_id(0);
    const int lid = get_local_id(0);
    const int L   = get_local_size(0);
    const int row = get_group_id(0);   

    if (row >= M) return;

    // -------- PASS 1: compute row max and streaming sum --------
    // local running values (private)
    float m = -INFINITY;    
    float s = 0.0f;         

    for (int start = 0; start < N; start += L) {
        int col = start + lid;

        float v = -INFINITY;
        if (col < N) v = S[row * (size_t)N + col] * scale;

        lbuf[lid] = v;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = L >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                float a = lbuf[lid];
                float b = lbuf[lid + stride];
                lbuf[lid] = fmax(a, b);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float tile_max = lbuf[0];

        float m_new = fmax(m, tile_max);
        float s_scaled = s * exp(m - m_new);

        float e = 0.0f;
        if (col < N) e = exp(v - m_new);
        lbuf[lid] = e;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = L >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                lbuf[lid] = lbuf[lid] + lbuf[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        float tile_sum = lbuf[0];

        s = s_scaled + tile_sum;
        m = m_new;

        barrier(CLK_LOCAL_MEM_FENCE);
    } 

    if (lid == 0) {
        lbuf[0] = m;
        if (L > 1) lbuf[1] = s;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float row_max = lbuf[0];
    float row_sum = (L > 1) ? lbuf[1] : s; 

    if (!isfinite(row_sum) || row_sum == 0.0f) {
        row_sum = 1e-37f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // -------- PASS 2: normalize and write outputs --------
    // walk tiles again and write normalized exp(v - row_max) / row_sum
    for (int start = 0; start < N; start += L) {
        int col = start + lid;
        if (col < N) {
            float v_orig = S[row * (size_t)N + col] * scale;
            float e = exp(v_orig - row_max);
            S[row * (size_t)N + col] = e / row_sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
