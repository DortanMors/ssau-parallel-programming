#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define SAFE_CALL(CallInstruction) { cudaError_t cudaError = CallInstruction; if (cudaError != cudaSuccess) { printf("CUDA error: %s at call %s", cudaGetErrorString(cudaError), #CallInstruction); exit(0); } }

__global__ void addKernel(double* c, double* a, double* b, unsigned int size) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

double calculateForParameters(int gridSize, int blockSize, int n, int maxTries) {
    double time = 0.0;
    for (int k = 0; k < maxTries; ++k) {
        int n2b = n * sizeof(double);

        double* a = (double*) calloc(n, sizeof(double));
        double* b = (double*) calloc(n, sizeof(double));
        double* c = (double*) calloc(n, sizeof(double));

        for (int i = 0; i < n; ++i) {
            a[i] = i;
            b[i] = i;
        }
        double* aDevice = NULL;
        double* bDevice = NULL;
        double* cDevice = NULL;
        SAFE_CALL(cudaMalloc((void**) &aDevice, n2b));
        SAFE_CALL(cudaMalloc((void**) &bDevice, n2b));
        SAFE_CALL(cudaMalloc((void**) &cDevice, n2b));
        cudaEvent_t start, stop;
        SAFE_CALL(cudaEventCreate(&start));
        SAFE_CALL(cudaEventCreate(&stop));
        SAFE_CALL(cudaMemcpy(aDevice, a, n2b, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaMemcpy(bDevice, b, n2b, cudaMemcpyHostToDevice));
        SAFE_CALL(cudaEventRecord(start, 0));
        addKernel <<< gridSize, blockSize >>> (cDevice, aDevice, bDevice, n);
        SAFE_CALL(cudaDeviceSynchronize());
        SAFE_CALL(cudaGetLastError());
        SAFE_CALL(cudaMemcpy(c, cDevice, n2b, cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaEventRecord(stop, 0));
        float gpuTime = 0.0f;
        SAFE_CALL(cudaDeviceSynchronize());
        SAFE_CALL(cudaEventElapsedTime(&gpuTime, start, stop));

        time += gpuTime * 1000;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(aDevice);
        cudaFree(bDevice);
        cudaFree(cDevice);
        free(a);
        free(b);
        free(c);
    }
    return time / maxTries;
}

double calculateConsistent(int n) {
    double* a = (double*) calloc(n, sizeof(double));
    double* b = (double*) calloc(n, sizeof(double));
    double* c = (double*) calloc(n, sizeof(double));

    for (int i = 0; i < n; ++i) {
        a[i] = i;
        b[i] = i;
    }

    auto start = std::chrono::system_clock::now();

    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }

    auto end = std::chrono::system_clock::now();
    double time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    free(a);
    free(b);
    free(c);
    return time;
}

int main(int argc, char* argv[]) {
    int maxTries = 12;
    int startN = 6000000;
    int nStep = 5;
    int gridsAndBlocks[6][2] = {
        { 2048, 1024 },
        { 1024, 1024 },
        { 2048, 512 },
        { 1024, 512 },
        { 2048, 256 },
        { 1024, 256 }
    };

    for (int n = startN; n > 240000 - 1 ; n /= nStep) {
        printf("n = %d\n", n);
        printf("Consistent: ");
        double cTime = 0.0;
        for (int k = 0; k < maxTries; ++k) {
            cTime += calculateConsistent(n);
        }
        printf("%.4f\n", cTime / maxTries);
        printf("CUDA: ");
        for (int j = 0; j < 6; ++j) {
            printf("[GridDim, BlockDim] = [%d, %d]: ", gridsAndBlocks[j][0], gridsAndBlocks[j][1]);
            double time = calculateForParameters(gridsAndBlocks[j][0], gridsAndBlocks[j][1], n, maxTries);
            printf("%.4f\n", time);
        }
    }
    return 0;
}
