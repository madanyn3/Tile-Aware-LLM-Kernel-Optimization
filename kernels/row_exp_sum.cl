// kernels/row_exp_sum.cl
// Compute the sum of exponentials of each row of a 2D matrix S (size MxN)
// and store the results in row_exp_sum_out (size M).

__kernel void row_exp_sum (
    const int M, const int N,
    __global float* S,
    __global const float* row_max,
    __global float* row_exp_sum_out)
{
    int row = get_global_id(0);
    if (row >= M) return;

    float sum_val;
    for (int i = 0; i < N; ++i) {
        float shifted = S[row * N + i] - row_max[row];
        float exp_val = exp(shifted);
        S[row * N + i] = exp_val;
        sum_val += exp_val;
    }
    row_exp_sum_out[row] = sum_val;
}