/******************************************************************************
 * FILE: omp_solved2.c
 * DESCRIPTION:
 *   Another OpenMP program with a bug. Fixed.
 * AUTHOR: Manyuan Tao
 * LAST REVISED: 03/20/17
 ******************************************************************************/

/*
 BUG1: change the date type of 'total' from float to double, for higher precision.
 BUG2: 'tid' was shared among all the threads, which caused a data race condition;
       set 'tid' a private variable for each thread.
 BUG3: 'total' was also shared and different threads accessed it without synchronization;
       we want to sum up all the numbers in the loop,
       so we can use a reduction on the shared variable 'total'.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[])
{
    int nthreads, i, tid;
    double total;   //!BUG1: change 'float' to 'double', for higher precision.
    
    /*** Spawn parallel region ***/
#pragma omp parallel private(tid)   //!BUG2: set 'tid' a private variable for each thread.
    {
        /* Obtain thread number */
        tid = omp_get_thread_num();
        /* Only master thread does this */
        if (tid == 0) {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
        printf("Thread %d is starting...\n",tid);
        
#pragma omp barrier
        
        /* do some work */
        total = 0.0;
#pragma omp for schedule(dynamic,10) reduction(+:total)   //!BUG3: 'total' needs reduction here.
        for (i=0; i<1000000; i++)
            total = total + i*1.0;
        
        printf ("Thread %d is done! Total= %e\n",tid,total);
        
    } /*** End of parallel region ***/
}
