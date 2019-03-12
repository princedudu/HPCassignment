/******************************************************************************
 * FILE: omp_solved5.c
 * DESCRIPTION:
 *   Using SECTIONS, two threads initialize their own array and then add
 *   it to the other's array, however a deadlock occurs. Fixed
 * AUTHOR: Manyuan Tao
 * LAST REVISED: 03/20/17
 ******************************************************************************/

/*
 BUG1: this was a classic deadlock problem where two locks were acquired by two different threads in reversed order. Now the threads are deadlocked: neither thread will give up its lock until it acquires the other lock, but neither thread will be able to acquire the other lock until the other thread gives it up.
 SOLUTION: either make all sections acquire the two locks in the same order,
           or unset the first lock before attempting for the second one (I use this here).
 BUG2: forget to destroy the locks after the parallelism finishes.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000000
#define PI 3.1415926535
#define DELTA .01415926535

int main (int argc, char *argv[])
{
    int nthreads, tid, i;
    float a[N], b[N];
    omp_lock_t locka, lockb;
    
    /* Initialize the locks */
    omp_init_lock(&locka);
    omp_init_lock(&lockb);
    
    /* Fork a team of threads giving them their own copies of variables */
#pragma omp parallel shared(a, b, nthreads, locka, lockb) private(tid)
    {
        
        /* Obtain thread number and number of threads */
        tid = omp_get_thread_num();
#pragma omp master
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d starting...\n", tid);
#pragma omp barrier
        
#pragma omp sections nowait
        {
#pragma omp section
            {
                printf("Thread %d initializing a[]\n",tid);
                omp_set_lock(&locka);
                for (i=0; i<N; i++)
                    a[i] = i * DELTA;
                omp_unset_lock(&locka);   //!BUG1: unset locka when finished writing to a.
                omp_set_lock(&lockb);
                printf("Thread %d adding a[] to b[]\n",tid);
                for (i=0; i<N; i++)
                    b[i] += a[i];
                omp_unset_lock(&lockb);
            }
            
#pragma omp section
            {
                printf("Thread %d initializing b[]\n",tid);
                omp_set_lock(&lockb);
                for (i=0; i<N; i++)
                    b[i] = i * PI;
                omp_unset_lock(&lockb);   //!BUG1: unset lockb when finished writing to b.
                omp_set_lock(&locka);
                printf("Thread %d adding b[] to a[]\n",tid);
                for (i=0; i<N; i++)
                    a[i] += b[i];
                omp_unset_lock(&locka);
                
            }
        }  /* end of sections */
    }  /* end of parallel region */
    
    /* Destroy the locks */
    omp_destroy_lock(&locka);   //!BUG2: forget to destroy the locks.
    omp_destroy_lock(&lockb);
    
}
