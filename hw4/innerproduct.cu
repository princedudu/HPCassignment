#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#define index(i, j, N)  ((i)*(N)) + (j)

// void reduction(double* sum_ptr, const double* a, long N){
//   double sum = 0;
//   #pragma omp parallel for schedule(static) reduction(+:sum)
//   for (long i = 0; i < N; i++) sum += a[i];
//   *sum_ptr = sum;
// }

void innerproduct(double* sum_ptr, const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i]*b[i];
  *sum_ptr = sum;
}

// void mat_vec_mul(double* sum_ptr, const double* a, const double* b, long N){
//   for(long i=0; i < N; i++){
//     double sum = 0;
//     #pragma omp parallel for schedule(static) reduction(+:sum)
//     for(long j=0; j < N; j++){
//       sum += a[index(i,j,N)]*b[j];
//     }
//     sum_ptr[i] = sum;
//   }
// }

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

__global__ void innerproduct_kernel(double* partial_c, const double* a, const double* b, long N){
  __shared__ double cache[BLOCK_SIZE];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int cacheidx = threadIdx.x;

  double tmp = 0;
  while(tid < N){
    tmp += a[tid]*b[tid];
    tid += blockDim.x * gridDim.x;
  }

  cache[cacheidx] = tmp;
  __syncthreads();

  int i=blockDim.x/2;
  while(i!=0){
    if(cacheidx<i)
      cache[cacheidx] += cache[cacheidx+i];
    __syncthreads();
    i/=2;
  }
  if(cacheidx==0)
    partial_c[blockIdx.x] = cache[0];
}

// //run matrix multiply vector
// int main() {
//   long N = 10000;

//   double *mat;
//   cudaMallocHost((void**)&mat, N * N * sizeof(double));
//   #pragma omp parallel for collapse(2) schedule(static) 
//   for (long i = 0; i < N; i++){
//     for (long j = 0; j < N; j++){
//       mat[index(i,j,N)] = (i+1.0)/(j+1);
//     }
//   }

//   double *v;
//   cudaMallocHost((void**)&v, N * sizeof(double));
//   #pragma omp parallel for schedule(static)
//   for (long i = 0; i < N; i++) v[i] = i+1;

//   // CPU version using openMP
//   double sum_ref;
//   double *result_v;
//   cudaMallocHost((void**)&result_v, N * sizeof(double));
//   double tt = omp_get_wtime();
//   for (long i = 0; i < N; i++){
//     double *x;
//     cudaMallocHost((void**)&x, N * sizeof(double));
//     #pragma omp parallel for
//     for (long j = 0; j < N; j++){
//       x[j] = mat[index(i,j,N)];
//     }
//     innerproduct(&sum_ref, x, v, N);
//     result_v[i] = sum_ref;
//     cudaFreeHost(x);
//   }
//   printf("CPU Bandwidth = %f GB/s\n", 2*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

//   double *vec_d;
//   cudaMalloc(&vec_d, N*sizeof(double));

//   double *result_cuda;
//   cudaMallocHost((void**)&result_cuda, N * sizeof(double));

//   cudaMemcpyAsync(vec_d, v, N*sizeof(double), cudaMemcpyHostToDevice);
//   cudaDeviceSynchronize();
//   tt = omp_get_wtime();

//   long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
//   for(long r=0; r<N; r++) {
//     double *x;
//     cudaMallocHost((void**)&x, N * sizeof(double));
//     #pragma omp parallel for
//     for (long j = 0; j < N; j++){
//       x[j] = mat[index(r,j,N)];
//     }
//     double* x_d;
//     cudaMalloc(&x_d, N*sizeof(double));
//     cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
//     double* pc_d;
//     cudaMalloc(&pc_d, Nb*sizeof(double));
//     double* partial_c;
//     cudaMallocHost((void**)&partial_c, Nb * sizeof(double));
//     innerproduct_kernel<<<Nb,BLOCK_SIZE>>>(pc_d, x_d, vec_d, N);
//     cudaMemcpyAsync(partial_c, pc_d, Nb*sizeof(double), cudaMemcpyDeviceToHost);
//     cudaDeviceSynchronize();
//     double sum=0;
//     for (long i = 0; i < Nb; ++i) {
//       sum += partial_c[i];
//     }
//     result_cuda[r] = sum;
//     cudaFreeHost(x);
//     cudaFreeHost(partial_c);
//     cudaFree(pc_d);
//     cudaFree(x_d);
//   }
//   printf("GPU Bandwidth = %f GB/s\n", 2*N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

//   // Error
//   double err = 0;
//   for (long i = 0; i < N; i++) err = std::max(err, std::abs(result_v[i] - result_cuda[i]));
//   printf("Error = %f\n", err);

//   cudaFreeHost(mat);
//   cudaFreeHost(v);
//   cudaFreeHost(result_v);
//   cudaFreeHost(result_cuda);
//   cudaFree(vec_d);
//   return 0;
// } 

// RUN inner product
int main() {
  long N = (1UL<<25);

  double *x;
  cudaMallocHost((void**)&x, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) x[i] = 1.0/(i+1);

  double *y;
  cudaMallocHost((void**)&y, N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) y[i] = i+1;

  double sum_ref;
  double tt = omp_get_wtime();
  innerproduct(&sum_ref, x, y, N);
  printf("CPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double sum = 0;
  double *x_d, *y_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&y_d, N*sizeof(double));

  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(y_d, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
  double* pc_d;
  cudaMalloc(&pc_d, Nb*sizeof(double));
  double* partial_c;
  cudaMallocHost((void**)&partial_c, Nb * sizeof(double));
  innerproduct_kernel<<<Nb,BLOCK_SIZE>>>(pc_d, x_d, y_d, N);
  cudaMemcpyAsync(partial_c, pc_d, Nb*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  for (int i = 0; i < Nb; ++i) {
		sum += partial_c[i];
	}

  printf("GPU Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("Error = %f\n", fabs(sum-sum_ref));

  cudaFree(x_d);
  cudaFree(y_d);
  cudaFree(pc_d);
  cudaFreeHost(x);
  cudaFreeHost(y);
  cudaFreeHost(partial_c);

  return 0;
}
