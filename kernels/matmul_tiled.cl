// kernels/matmul_tiled.cl
// Simple tiled matrix multiplication kernel

__kernel void matmul_tiled(
    const int M, const int N, const int K,
    __global const float* A, const int lda,
    __global const float* B, const int ldb,
    __global float* C, const int ldc )
{

    const int TILE = 16;

    int col = get_global_id(0);
    int row = get_global_id(1);

    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local float Asub[TILE][TILE];
    __local float Bsub[TILE][TILE];

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + lx;
        int bRow = t * TILE + ly;

        if (row < M && aCol < K)
            Asub[ly][lx] = A[row * lda + aCol];
        else
            Asub[ly][lx] = 0.0f;
        
        if (bRow < K && col < N)
            Bsub[ly][lx] = B[bRow * ldb + col];
        else
            Bsub[ly][lx] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE; ++k) {
            acc += Asub[ly][k] * Bsub[k][lx];
        }
        // Synchronize before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N) {
        C[row * ldc + col] = acc;
    }

}