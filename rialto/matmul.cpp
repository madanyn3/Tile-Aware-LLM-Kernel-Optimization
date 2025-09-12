#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

// simple matrix multiplication
// A and B are stored in row-major order
// C is stored in row-major order
// A is of size rowA x colA
// B is of size colA x colB

template <typename Ts, typename Td>
void 
matmul_simple (const Ts* A, const Ts* B, Td* C, unsigned int rowA, unsigned int colA, unsigned int colB) {
    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            Td acc = static_cast<Ts>(0);
            for (int k = 0; k < colA; k++) {
                acc += ( (*(A + i*colA + k)) * (*(B + k*colB + j)) )
            }
            *(C + i*colB + j) = acc;
            acc = static_cast<Ts>(0);
        }
    } 
}

template <typename Ts>
void
matrix_add (const Ts* A, const Ts* B, Ts* C, unsigned int row, unsigned int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            *(C + i*col + j) = ( *(A + i*col + j) + *(B + i*col + j) );
        }
    }
}

// blocked matrix multiplication
// A and B are stored in block-row-major order
// C is stored in block-row-major order
template <typename Ts, typename Td, unsigned int M, unsigned int K, unsigned int N>
void
matmul_tiled (const Ts* A, const Ts* B, Td* C, unsigned int rowA, unsigned int colA, unsigned int colB) {
    static_assert( ((rowA % M == 0) && (colA % K == 0) && (colB % N == 0)), "imperfect tiling not supported" );

    const unsigned int numTilesRowA = rowA / M;
    const unsigned int numTilesColA = colA / K;
    const unsigned int numTilesColB = colB / N;

    for (int i = 0; i < numTilesRowA; i++) {
        for (int j = 0; j < numTilesColA; j++) {
            Td* l_C = (C + (i*numTilesColB+j) * (M*N));
            Td l_tempC[M * N] = {0};
            for (int k = 0; k < numTilesColA; k++) {
                Ts* l_A = (A + (i*numTilesColA+k) * (M*K)); 
                Ts* l_B = (B + (k*numTilesColB+j) * (K*N));
                matmul_simple<Ts, Td>(l_A, l_B, &l_tempC, M, K, N);
                matrix_add<Td>(l_C, &l_tempC, l_C, M, N);
            }
        }
    }
}

// blocked matrix multiplication with multithreading
// A and B are stored in block-row-major order
// C is stored in block-row-major order
// each block C(i,j) is computed by a single thread
template <typename Ts, typename Td, unsigned int M, unsigned int K, unsigned int N>
void
matmul_tiled_multithreaded (const Ts* A, const Ts* B, Td* C, unsigned int rowA, unsigned int colA, unsigned int colB) {
    static_assert( ((rowA % M == 0) && (colA % K == 0) && (colB % N == 0)), "imperfect tiling not supported" );

    const unsigned int numTilesRowA = rowA / M;
    const unsigned int numTilesColA = colA / K;
    const unsigned int numTilesColB = colB / N;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numTilesRowA; i++) {
        for (int j = 0; j < numTilesColA; j++) {
            Td* l_C = (C + (i*numTilesColB+j) * (M*N));
            Td l_tempC[M * N] = {0};
            for (int k = 0; k < numTilesColA; k++) {
                Ts* l_A = (A + (i*numTilesColA+k) * (M*K)); 
                Ts* l_B = (B + (k*numTilesColB+j) * (K*N));
                matmul_simple<Ts, Td>(l_A, l_B, &l_tempC, M, K, N);
                matrix_add<Td>(l_C, &l_tempC, l_C, M, N);
            }
        }
    }
}