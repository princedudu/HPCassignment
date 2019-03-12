// The OpenMP version of Gauss-Seidel method with red-black coloring

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "util.h"
#ifdef _OPENMP
#include <omp.h>
#endif

double residual(int N, double h, double *f, double *u);

int main(int argc, char *argv[])
{
    int i, j, N;
    int iter, max_iters;
    double h, res, res_init;
    double *f, *u;
    
    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iters);
    h = 1./(N + 1);
    
#ifdef _OPENMP
    omp_set_num_threads(4);
# pragma omp parallel
    {
        int my_threadnum = omp_get_thread_num();
        int numthreads = omp_get_num_threads();
        printf("Hello, I'm thread %d out of %d.\n", my_threadnum, numthreads);
    }
#else
    printf("Hello, I'm process %d out of %d.\n", 0, 1);
#endif
    
    /* allocation of vectors, including boundary ghost points */
    f = (double *) malloc(sizeof(double) * pow(N+2, 2));
    u = (double *) malloc(sizeof(double) * pow(N+2, 2));
    
    /* fill vectors */
    for (i = 0; i < pow(N+2, 2); i++) {
        f[i] = 1;
        u[i] = 0;
    }
    
    /* compute initial residual */
    res_init = residual(N, h, f, u);
    res = res_init;
    
    /* timing */
    timestamp_type time1, time2;
    get_timestamp(&time1);
    
    for (iter = 1; iter <= max_iters && res_init/res < 1e4; iter++) {
        
        /* Gauss-Seidel method with red-black coloring */
        
#pragma omp parallel for private(j)
        /* update black points first */
        for (i = 1; i <= N; i++)
            for (j = 1 + i % 2; j <= N; j += 2)
                u[i*(N+2)+j]= 0.25 * (pow(h, 2) * f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)]);
        
#pragma omp parallel for private(j)
        /* update red points next */
        for (i = 1; i <= N; i++)
            for (j = 2 - i % 2; j <= N; j += 2)
                u[i*(N+2)+j]= 0.25 * (pow(h, 2) * f[i*(N+2)+j] + u[(i-1)*(N+2)+j] + u[i*(N+2)+(j-1)] + u[(i+1)*(N+2)+j] + u[i*(N+2)+(j+1)]);
        
        /* compute residual for each iteration */
        res = residual(N, h, f, u);
        printf("Iteration: %d;\t Residual: %f.\n", iter, res);
    }
    
    /* timing */
    get_timestamp(&time2);
    double elapsed = timestamp_diff_in_seconds(time1, time2);
    printf("Time elapsed is %f seconds.\n", elapsed);
    
    /* clean up */
    free(f);
    free(u);
    
    return 0;
}


/* define funtion to compute residual */
double residual(int N, double h, double *f, double *u)
{
    int i, j;
    double sum, residual;
    residual = 0;
    
#pragma omp parallel for private(j,sum) reduction(+:residual)
    for (i = 1; i <= N; i++)
        for (j = 1; j <= N; j++){
            sum = (4 * u[i*(N+2)+j] - u[(i-1)*(N+2)+j] - u[i*(N+2)+(j-1)] - u[(i+1)*(N+2)+j] - u[i*(N+2)+(j+1)]) / pow(h, 2);
            residual += pow(sum - f[i*(N+2)+j], 2);
        }
    return sqrt(residual);
}
