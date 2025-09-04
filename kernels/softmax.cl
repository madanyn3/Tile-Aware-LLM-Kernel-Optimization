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
