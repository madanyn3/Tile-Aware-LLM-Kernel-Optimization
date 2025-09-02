// kernels/vec_add.cl
// simple test for system setup sanity

__kernel void vec_add(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int n) 
{
    int gid = get_global_id(0);
    if (gid < n) {
        C[gid] = A[gid] + B[gid];
    }

}