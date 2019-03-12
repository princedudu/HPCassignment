/******************************************************************************
 * FILE: omp_solved4.c
 * DESCRIPTION:
 *   This very simple program causes a segmentation fault. Fixed
 * AUTHOR: Manyuan Tao
 * LAST REVISED: 03/20/17
 ******************************************************************************/

/*
 BUG: the stack for each thread doesn't have enough space to hold a double array of size 1048*1048, i.e., stack overflow, thus causing a segmentation fault.
 SOLUTION1: change N to a smaller number, e.g., N=512 works fine.
 SOLUTION2: allocate the memory of the huge array on heap (I use this here).
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[])
{
    int nthreads, tid, i, j;
    double **a;   //!BUG: use allocated memory instead
    
    /* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid,a)
    {
        /* Allocate an array on heap for each thread */
        a = (double **) malloc(sizeof(double*) * N);
        for (i=0; i<N; i++)
            a[i] = (double *) malloc(sizeof(double) * N);
        
        /* Obtain/print thread info */
        tid = omp_get_thread_num();
        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d starting...\n", tid);
        
        /* Each thread works on its own private copy of the array */
        for (i=0; i<N; i++)
            for (j=0; j<N; j++)
                a[i][j] = tid + i + j;
        
        /* For confirmation */
        printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);
        
        free(a);
        
    }  /* All threads join master thread and disband */
    
}
