#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

#define index(i, j, N)  ((i)*(N)) + (j)
#define BLOCK_SIZE 1024

// get nearby value(left, right, up, down) of k-th u at point (i,j)
__device__ void getNearbyValue(int i,int j, int N, double& left, double& right, double& up, double& down, double* u){
    // first row
    if(i==0){
        if(j==0){
            left = 0;
            right = u[index(i,j+1,N)];
        }
        else if (j==N-1) {
            left = u[index(i,j-1,N)];
            right = 0;
        }
        else{
            left = u[index(i,j-1,N)];
            right = u[index(i,j+1,N)];
        }
        down = 0;
        up = u[index(i+1,j,N)];
    }
    // last row
    if (i==N-1){
        if(j==0){
            left = 0;
            right = u[index(i,j+1,N)];
        }
        else if (j==N-1) {
            left = u[index(i,j-1,N)];
            right = 0;
        }
        else{
            left = u[index(i,j-1,N)];
            right = u[index(i,j+1,N)];
        }
        up = 0;
        down = u[index(i-1,j,N)];        
    }
    // middle row
    else{
        if(j==0){
            left = 0;
            right = u[index(i,j+1,N)];
        }
        else if (j==N-1) {
            left = u[index(i,j-1,N)];
            right = 0;
        }
        else{
            left = u[index(i,j-1,N)];
            right = u[index(i,j+1,N)];
        }
        up = u[index(i+1,j,N)]; 
        down = u[index(i-1,j,N)];
    }
}

__device__ void indextoij(int index,int N,int& i, int& j){
    i=index/N;    
    j=index%N;
}

__global__ void jacobi2D_kernel(double* u_next,double* f, double* u, int N, double h){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<N*N){
        int i,j=0;
        indextoij(tid,N,i,j);
        // printf("%d %d\n",i,j);
        double left,right,up,down;
        getNearbyValue(i,j,N,left,right,up,down,u);
        u_next[tid]=(pow(h,2)*f[index(i,j,N)]+up+down+left+right)/4.0;
    }
}

void jacobi2D(double* f, double* u, int N, int max_iteration){
    double h = 1.0/(N+1.0);
    double *u_next = (double*) malloc(N * N * sizeof(double)); // (k+1)-th u
    memcpy(u_next, u, N * N * sizeof(double));

    double *f_d;
    cudaMalloc(&f_d, N*N*sizeof(double));
    cudaMemcpy(f_d, f, N*N*sizeof(double), cudaMemcpyHostToDevice);

    double *u_d;
    cudaMalloc(&u_d, N*N*sizeof(double));
    cudaMemcpy(u_d, u, N*N*sizeof(double), cudaMemcpyHostToDevice);

    double *u_next_d;
    cudaMalloc(&u_next_d, N*N*sizeof(double));
    cudaMemcpy(u_next_d, u_next, N*N*sizeof(double), cudaMemcpyHostToDevice);    
    long Nb = (N*N+BLOCK_SIZE-1)/(BLOCK_SIZE);    
    // printf("%d %d\n",Nb,BLOCK_SIZE);
    for(int k=0;k<max_iteration;k++){    
        jacobi2D_kernel<<<Nb,BLOCK_SIZE>>>(u_next_d,f_d,u_d,N,h);

        cudaMemcpy(u_d, u_next_d, N*N*sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(u, u_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);    

    cudaFree(f_d);
    cudaFree(u_d);
    cudaFree(u_next_d);
    free(u_next);
}

int main(int argc, char * argv[]){
    int maxite=5000;
    int N;
    if(argc!=2){
        fprintf(stderr, "usage: jacobi2D-omp <N>\n");
        exit(1);
    }
    N = atoi(argv[1]);

    double* u = (double*) malloc(N * N * sizeof(double)); // solution
    double* f = (double*) malloc(N * N * sizeof(double)); // RHS

    for(int i=0;i<N*N;i++){
        u[i] = 0.0; // initial guess
        f[i] = 1.0; // right hand side equals 1
    }
    // printf("Max thread number: %d\n", omp_get_max_threads()); // => 64
    double tt = omp_get_wtime();
    jacobi2D(f,u,N,maxite);
    double time = omp_get_wtime()-tt;
    printf("\n");
    printf("Total time taken: %10f seconds\n", time);
    
    // for(int i=0;i<N*N;i++){
    //     printf("%lf\n", u[i]);
    // }

    free(u);
    free(f);
    return 0;
}
