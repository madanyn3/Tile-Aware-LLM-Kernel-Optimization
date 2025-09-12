// kernels/scale_inplace.cl
// Scale each element of a 2D matrix S (size MxN) by a given factor in place.

__kernel void scale_inplace (
    const int M, const int N,
    const float scale,
    __global float* S)
{
    int row = get_global_id(0);
    if (row >= M) return;

    for (int i = 0; i < N; ++i) {
        S[row * N + i] *= scale;
    }
}