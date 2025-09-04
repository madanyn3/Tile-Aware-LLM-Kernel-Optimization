// kernels/attention_naive.cl
// Naive implementation of the attention mechanism using OpenCL.
// This kernel computes the attention scores for a batch of sequences.

inline float row_max (
    const int row, const int N,
    __global const float* S)
{
    float row_max = -FLT_MAX
    for (int i = 0; i < N; ++i) {
        float val = S[row * N + i];
        if (val > row_max) row_max = val;
    }
    return row_max;
}

inline float row_exp_sum (
    const int row, const int N,
    __global const float* S,
    const float row_max)
{
    float row_sum_exp;
    for (int i = 0; i < N; ++i) {
        float shifted = S[row * N + i] - row_max;
        float exp_val = exp(shifted);
        S[row * N + i] = exp_val
        row_sum_exp += exp_val;
    }
    return row_sum_exp;
}

__kernel void attention_naive (
    const int M, const int N,
    __global float* S)
{
    int row = get_global_id(0);
    if (row >= M) return;
}