// kernels/row_max.cl
// Compute the maximum value in each row of a 2D matrix S (size MxN)
// and store the results in row_max_out (size M).

__kernel void row_max (
    const int M, const int N,
    __global const float* S,
    __global float row_max_out)
{
    row = get_global_id(0)
    if (row > M) return;

    float max_val = -FLT_MAX
    for (int i = 0; i < N; ++i) {
        float val = S[row * N + i];
        if (val > max_val) max_val = val;
    }
    row_max_out[row] = max_val;
}