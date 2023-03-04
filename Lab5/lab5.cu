%%cuda --name lab5curand.cu

#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <ctime>

#define SAFE_CALL(CallInstruction) { cudaError_t cudaError = CallInstruction; if (cudaError != cudaSuccess) { printf("CUDA error: %s at call %s", cudaGetErrorString(cudaError), #CallInstruction); exit(0); } }
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", matrix[i + j * rows]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrix_mult(float* A, float* B, float* C, int N)
{
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            C[IDX2C(i, j, N)] = 0;
            for (k = 0; k < N; k++) {
                C[IDX2C(i, j, N)] += A[i + k * N] * B[k + j * N];
            }
        }
    }
}

void fill_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i + j * rows] = (rand() + 0.0) / RAND_MAX;
        }
    }
}

void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

double calculateGPU(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start, 0));
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    SAFE_CALL(cudaEventRecord(stop, 0));

    cublasDestroy(handle);

    float gpuTime = 0.0f;
    SAFE_CALL(cudaEventElapsedTime(&gpuTime, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double time = gpuTime * 1000;
    return time;
}

void testGPU(int squareSize) {
// Заполнение матриц на GPU
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = squareSize;

    float *d_A, *d_B, *d_C, *C;

    SAFE_CALL(cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float)));
    SAFE_CALL(cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float)));
    SAFE_CALL(cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(float)));
    C = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));

    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));
    SAFE_CALL(cudaEventRecord(start, 0));
    GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
    GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);
    SAFE_CALL(cudaMemcpy(c, d_C, nr_rows_C * nr_cols_C * sizeof(float), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());
    SAFE_CALL(cudaEventRecord(stop, 0));
    float gpuTime = 0.0f;
    SAFE_CALL(cudaEventElapsedTime(&gpuTime, start, stop));
    cout << "GPU create: " << gpuTime * 1000 << " milliseconds\n";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double gpuTime = calculateGPU(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    cout << "calculateGPU time = " << gpuTime << " microseconds";
    print_matrix(C, nr_rows_C, nr_cols_C);

    free(C);
    SAFE_CALL(cudaFree(d_A));
    SAFE_CALL(cudaFree(d_B));
    SAFE_CALL(cudaFree(d_C));
}

void testCPU(int squareSize) {
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = squareSize;
    float *d_A, *d_B;
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    SAFE_CALL(cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(float)));
    SAFE_CALL(cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(float)));

    double start_consistent = clock();
    fill_matrix(h_A, nr_rows_A, nr_cols_A);
    fill_matrix(h_B, nr_rows_B, nr_cols_B);
    SAFE_CALL(cudaMemcpy(h_A, d_B, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaMemcpy(h_B, d_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyDeviceToHost));
    SAFE_CALL(cudaDeviceSynchronize());
    double cpuGenerateTime = ((clock() - start_consistent) / CLOCKS_PER_SEC) * 1000 * 1000;
    cout << "GPU create: " << cpuGenerateTime <<  " milliseconds\n";

    start_consistent = clock();
    matrix_mult(h_A, h_B, h_C, nr_rows_A);
    double cpuTime = ((clock() - start_consistent) / CLOCKS_PER_SEC) * 1000 * 1000;
    cout << "calculateCPU time: " << cpuGenerateTime << " milliseconds\n";
    print_matrix(h_C, nr_rows_C, nr_cols_C);

    free(h_A);
    free(h_B);
    free(h_C);

    SAFE_CALL(cudaFree(d_A));
    SAFE_CALL(cudaFree(d_B));
}

int main() {
    for (int size = 100; size <= 2500; size *= 500) {
        testCPU(size);
        testGPU(size);
    }
    return 0;
}
