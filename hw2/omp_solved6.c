/******************************************************************************
 * FILE: omp_solved6.c
 * DESCRIPTION:
 *   This program compiles and runs fine, but produces the wrong result. Fixed.
 * AUTHOR: Manyuan Tao
 * LAST REVISED: 03/20/17
 ******************************************************************************/

/*
 BUG: 'sum' was redeclared in each dotprod call and was therefore a private variable inside the function, however reduction must operate on shared variables.
 SOLUTION: replace the 'sum' declerations in main() and dotprod() with a global variable.
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

/* Define sum as a global variable */
float sum;

float dotprod ()
{
    int i,tid;
    //float sum;   //!BUG: 'sum' redeclared here made it a private variable, conflicting with reduction.
    
    tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
    for (i=0; i < VECLEN; i++)
    {
        sum = sum + (a[i]*b[i]);
        printf("tid=%d i=%d\n",tid,i);
    }
}


int main (int argc, char *argv[]) {
    int i;
    //float sum;   //!BUG: change the scope of sum to be global.
    
    for (i=0; i < VECLEN; i++)
        a[i] = b[i] = 1.0 * i;
    sum = 0.0;
    
#pragma omp parallel shared(sum)
    dotprod();
    
    printf("Sum = %f\n",sum);
    
}
