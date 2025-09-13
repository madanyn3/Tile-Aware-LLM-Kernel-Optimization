#include<vector>
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include<cstdint>
#include<chrono>

// simple matrix multiplication
// A and B are stored in row-major order
// C is stored in row-major order
// A is of size rowA x colA
// B is of size colA x colB

template <typename Ts, typename Td>
void 
matmul_simple (const Ts* A, const Ts* B, Td* C, unsigned int rowA, unsigned int colA, unsigned int colB) {
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
            Td acc = static_cast<Ts>(0);
            for (int k = 0; k < colA; k++) {
                acc += ( (*(A + i*colA + k)) * (*(B + k*colB + j)) );
            }
            *(C + i*colB + j) = acc;
            acc = static_cast<Ts>(0);
        }
    }
    end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("Time to compute simple matmul: %lu s\n", diff.count()); 
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

// 16x16 tile FP32 matmul microkernel with avx512 instructions
// A and B are stored in row-major order
// C is stored in row-major order

#include<immintrin.h>
#include<cstddef>
#include<cstdint>

void
matmul_16x16_microkernel_fp32 (
    const float* A, const float* B, float* C,
    std::size_t lda = 16, std::size_t ldb = 16, std::size_t ldc = 16) {

    auto start = std::chrono::high_resolution_clock::now();

    __m512 c0 = _mm512_loadu_ps(C + 0*ldc);
    __m512 c1 = _mm512_loadu_ps(C + 1*ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2*ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3*ldc);
    __m512 c4 = _mm512_loadu_ps(C + 4*ldc);
    __m512 c5 = _mm512_loadu_ps(C + 5*ldc);
    __m512 c6 = _mm512_loadu_ps(C + 6*ldc);
    __m512 c7 = _mm512_loadu_ps(C + 7*ldc);
    __m512 c8 = _mm512_loadu_ps(C + 8*ldc);
    __m512 c9 = _mm512_loadu_ps(C + 9*ldc);
    __m512 c10 = _mm512_loadu_ps(C + 10*ldc);
    __m512 c11 = _mm512_loadu_ps(C + 11*ldc);
    __m512 c12 = _mm512_loadu_ps(C + 12*ldc);
    __m512 c13 = _mm512_loadu_ps(C + 13*ldc);
    __m512 c14 = _mm512_loadu_ps(C + 14*ldc);
    __m512 c15 = _mm512_loadu_ps(C + 15*ldc);

    for (uint8_t k = 0; k < 16; k++) {
        const float *b_row = B + k*ldb;
        __m512 bvec = _mm512_loadu_ps(b_row);

        __m512 a0 = _mm512_set1_ps(A[0*lda + k]);
        c0 = _mm512_fmadd_ps(a0, bvec, c0);

        __m512 a1 = _mm512_set1_ps(A[1*lda + k]);
        c1 = _mm512_fmadd_ps(a1, bvec, c1);

        __m512 a2 = _mm512_set1_ps(A[2*lda + k]);
        c2 = _mm512_fmadd_ps(a2, bvec, c2);

        __m512 a3 = _mm512_set1_ps(A[3*lda + k]);
        c3 = _mm512_fmadd_ps(a3, bvec, c3);

        __m512 a4 = _mm512_set1_ps(A[4*lda + k]);
        c4 = _mm512_fmadd_ps(a4, bvec, c4);

        __m512 a5 = _mm512_set1_ps(A[5*lda + k]);
        c5 = _mm512_fmadd_ps(a5, bvec, c5);

        __m512 a6 = _mm512_set1_ps(A[6*lda + k]);
        c6 = _mm512_fmadd_ps(a6, bvec, c6);

        __m512 a7 = _mm512_set1_ps(A[7*lda + k]);
        c7 = _mm512_fmadd_ps(a7, bvec, c7);

        __m512 a8 = _mm512_set1_ps(A[8*lda + k]);
        c8 = _mm512_fmadd_ps(a8, bvec, c8);

        __m512 a9 = _mm512_set1_ps(A[9*lda + k]);
        c9 = _mm512_fmadd_ps(a9, bvec, c9);

        __m512 a10 = _mm512_set1_ps(A[10*lda + k]);
        c10 = _mm512_fmadd_ps(a10, bvec, c10);

        __m512 a11 = _mm512_set1_ps(A[11*lda + k]);
        c11 = _mm512_fmadd_ps(a11, bvec, c11);

        __m512 a12 = _mm512_set1_ps(A[12*lda + k]);
        c12 = _mm512_fmadd_ps(a12, bvec, c12);

        __m512 a13 = _mm512_set1_ps(A[13*lda + k]);
        c13 = _mm512_fmadd_ps(a13, bvec, c13);

        __m512 a14 = _mm512_set1_ps(A[14*lda + k]);
        c14 = _mm512_fmadd_ps(a14, bvec, c14);

        __m512 a15 = _mm512_set1_ps(A[15*lda + k]);
        c15 = _mm512_fmadd_ps(a15, bvec, c15);
    }

    _mm512_storeu_ps(C + 0*ldc, c0);
    _mm512_storeu_ps(C + 1*ldc, c1);
    _mm512_storeu_ps(C + 2*ldc, c2);
    _mm512_storeu_ps(C + 3*ldc, c3);
    _mm512_storeu_ps(C + 4*ldc, c4);
    _mm512_storeu_ps(C + 5*ldc, c5);
    _mm512_storeu_ps(C + 6*ldc, c6);
    _mm512_storeu_ps(C + 7*ldc, c7);
    _mm512_storeu_ps(C + 8*ldc, c8);
    _mm512_storeu_ps(C + 9*ldc, c9);
    _mm512_storeu_ps(C + 10*ldc, c10);
    _mm512_storeu_ps(C + 11*ldc, c11);
    _mm512_storeu_ps(C + 12*ldc, c12);
    _mm512_storeu_ps(C + 13*ldc, c13);
    _mm512_storeu_ps(C + 14*ldc, c14);
    _mm512_storeu_ps(C + 15*ldc, c15);

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("Time to compute 16x16 microkernel: %lu s\n", diff.count());

}

// same microkernel but compact code
void
matmul_16x16_microkernel_fp32_compact (
    const float* A, const float* B, float* C,
std::size_t lda = 16, std::size_t ldb = 16, std::size_t ldc = 16) {

    auto start = std::chrono::high_resolution_clock::now();

    __m512 tempC[ldc];
    #pragma loop(ivdep)
#pragma loop(unroll)
    for(uint8_t i = 0; i < ldc; i++) {
        tempC[i] = _mm512_loadu_ps(C + i*ldc);
    }

    for (uint8_t k = 0; k < ldc; k++) {
        const float* brow = B + k*ldb;
        __m512 bvec = _mm512_loadu_ps(brow);

        #pragma loop(ivdep)
#pragma loop(unroll)
        for (uint8_t i = 0; i < ldc; i++) {
            __m512 ai = _mm512_set1_ps(A[i*lda + k]);
            tempC[i] = _mm512_fmadd_ps(ai, bvec, tempC[i]);
        }
    }

    #pragma loop(ivdep)
#pragma loop(unroll)
    for (uint8_t i = 0; i < ldc; i++) {
        _mm512_storeu_ps(C + i*ldc, tempC[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    printf("Time to compute 16x16 compact microkernel: %lu s\n", diff.count());
}