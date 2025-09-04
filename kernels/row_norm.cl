// kernels/row_norm.cl
// Normalize each row of a 2D matrix S (size MxN) by dividing each element
// by the corresponding row sum stored in row_exp_sum (size M).

__kernel void row_norm (
    const int M, const int N,
    __global float* S,
    __global const float* row_sum)
{
    int row = get_global_id(0);
    if (row >= M) return;

    float inv_sum = 1.0f / row_sum[row];
    for (int i = 0; i < N; ++i) {
        S[row * N + i] *= inv_sum;
    }
}