%%cu

#include <cstdlib>
#include <curand.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <chrono>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

void matrixMult(const float* A, const float* B, float* C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    double elapsedTime, allTime = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < 12; i++) {
        cudaEventRecord(start, 0);
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        cudaEventRecord(stop, 0);
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&elapsedTime, start, stop);
        allTime += elapsedTime * 1000;
    }
    printf("\nAverage parallel time: %f\n", allTime / 12);
    cublasDestroy(handle);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {    
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

void consistent(const float* A, const float* B, float* C, const int m, const int k, const int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C[IDX2C(i, j, n)] = 0.0;
            for (int r = 0; r < n; ++r) {
                C[IDX2C(i, j, n)] += A[IDX2C(i, r, k)] * B[IDX2C(r, j, n)];
            }
        }
    }
}

void print_matrix(float* matrix, int rows, int cols) {
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
          printf("%f ", matrix[IDX2C(i, j, 3)]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    printf("\nFloat, (100, 500, 2500), (с Т, c Т)\n");
    for (int n = 100; n <= 2500; n *= 5) {
        int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
        nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = n;
        float *h_A = (float*)malloc(nr_rows_A * nr_cols_A * sizeof(float));
        float *h_B = (float*)malloc(nr_rows_B * nr_cols_B * sizeof(float));
        float *h_C = (float*)malloc(nr_rows_C * nr_cols_C * sizeof(float));

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float));
        cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float));
        cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float));
        double t = 0;
        for (int step=0; step<12; ++step) {
            srand(time(0));
            auto start = std::chrono::system_clock::now();
            for (int i = 0; i < nr_rows_A * nr_rows_A; i++) {
                h_A[i] = (float)rand()/RAND_MAX;
                h_B[i] = (float)rand()/RAND_MAX;
            }
            cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);
            GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
            GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
            cudaMemcpy(h_A, d_A, nr_rows_A * nr_cols_A*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_B, d_B, nr_rows_B * nr_cols_B*sizeof(float), cudaMemcpyDeviceToHost);
            matrixMult(h_A, h_B, h_C, nr_rows_A, nr_cols_A, nr_cols_B);
            cudaMemcpy(h_C, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost);

            auto end = std::chrono::system_clock::now();
            double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            t += time;
        }
        printf("\nn = %d\n", n);
        printf("A:\n");
        print_matrix(h_A, nr_rows_A, nr_cols_A);
        printf("B:\n");
        print_matrix(h_B, nr_rows_B, nr_cols_B);
        printf("C:\n");
        print_matrix(h_C, nr_rows_C, nr_cols_C);
        printf("Time %f\n ", t/12);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
    }
    return 0;
}