%%cuda --name my_curand.cu

#include <cstdlib>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <ctime>

using namespace std;


void multiply(float* A, float* B, float* C, int N)
{
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C[i + j * N] = 0;
      for (k = 0; k < N; k++)
        C[i + j * N] += A[i + k * N] * B[k + j * N];
    }
  }
}


void GPU_fill_rand(float *A, int nr_rows_A, int nr_cols_A) {
  curandGenerator_t prng;
  curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());
  curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

// C(m,n) = A(m,k) * B(k,n)
void gpu_blas_mmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
  int lda=m,ldb=k,ldc=m;
  const float alf = 1;
  const float bet = 0;
  const float *alpha = &alf;
  const float *beta = &bet;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  cublasDestroy(handle);
}

void print_matrix(float* matrix, int rows, int cols) {
  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j){
      printf("%f ", matrix[i + j * rows]);
    }
    printf("\n");
  }
  printf("\n");
}

void fill_matrix(float* matrix, int rows, int cols) {
  for(int i = 0; i < rows; ++i){
    for(int j = 0; j < cols; ++j) {
      matrix[i + j * rows] = (rand() + 0.0) / RAND_MAX;
    }
  }
}



int main() {
  int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
  nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = 6000;
  float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
  float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
  float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
  float *h_C_ = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
  cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
  cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));


  GPU_fill_rand(d_A, nr_rows_A, nr_cols_A);
  GPU_fill_rand(d_B, nr_rows_B, nr_cols_B);


  fill_matrix(h_A, nr_rows_A, nr_cols_A);
  fill_matrix(h_B, nr_rows_B, nr_cols_B);

  double start = clock();
  cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(float), cudaMemcpyHostToDevice);
  double end = clock();

  cudaMemcpy(h_A, d_A, nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);


  printf("A = \n");
  print_matrix(h_A, 5, 5);

  printf("B = \n");
  print_matrix(h_B, 5, 5);

  ///////////////////////////
  double start_ = clock();
  gpu_blas_mmul(d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
  //multiply(h_A, h_B, h_C_, nr_rows_A);
  double end_ = clock();

  cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);
  printf("C = \n");
  print_matrix(h_C, 5, 5);

  //cout << "\nParallel mul time = " << (end_ - start_) / CLOCKS_PER_SEC << endl;
  cout << "\nSendRecv time = " << (end - start) / CLOCKS_PER_SEC << endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);
  return 0;
}
