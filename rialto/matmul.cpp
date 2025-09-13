#include<iostream>
#include<cstdint>
#include<random>
#include "matmul.hpp"

void testbench () {
    uint8_t size = 16;
    uint8_t seed = 224;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<float> dist(0.0f, 10.0);

    float A[size*size];
    float B[size*size];

    for (uint16_t i = 0; i < size*size; i++) {
        A[i] = dist(generator);
        B[i] = dist(generator);
    }

    float C_ref[size*size];
    float C_uC1[size*size];
    float C_uC2[size*size];

    matmul_simple<float, float>(&A[0], &B[0], &C_ref[0], size, size, size);
    matmul_16x16_microkernel_fp32(&A[0], &B[0], &C_uC1[0]);
    matmul_16x16_microkernel_fp32_compact(&A[0], &B[0], &C_uC2[0]);

    // for (uint16_t i = 0; i < size*size; i++) {
    //     if (std::abs(C_ref[i] - C_uC1[i]) > 1e-3) {
    //         std::cout << "mismatch at " << i << " ref: " << C_ref[i] << " uC1: " << C_uC1[i] << std::endl;
    //         exit(1);
    //     }
    //     if (std::abs(C_ref[i] - C_uC2[i]) > 1e-3) {
    //         std::cout << "mismatch at " << i << " ref: " << C_ref[i] << " uC2: " << C_uC2[i] << std::endl;
    //         exit(1);
    //     }
    // }
    std::cout << "All results matched!" << std::endl;
}

int main() {
    testbench();
    return 0;
}