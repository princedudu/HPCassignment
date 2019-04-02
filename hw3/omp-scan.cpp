#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <iostream>

#define THREAD_NUM 8

using namespace std;

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  #pragma omp parallel num_threads(THREAD_NUM)
  {
    int tid = omp_get_thread_num();
    long psize = n/THREAD_NUM; // part size assign to each thread 
    long start_idx = psize*tid;
    long end_idx = start_idx + psize;
    if(tid == THREAD_NUM-1){ // for the last thread, take care all remainings
      end_idx = n;
    }
    // #pragma omp critical
    // {
    //   cout<<"tid:"<<tid<<" start:"<<start_idx<<" end:"<<end_idx<<endl;
    // }
    if(tid == 0)
      prefix_sum[start_idx] = 0;
    else
      prefix_sum[start_idx] = A[start_idx-1]; // need correction prefix_sum[start_idx-1];
    for(long i=start_idx+1; i<end_idx; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }

    #pragma omp barrier

    long correction = 0;
    if(tid!=0){ // except first part, require correction
      // calculate correction
      for(int i=start_idx-1; i>0; i-=psize)
        correction += prefix_sum[i];
    }

    #pragma omp barrier

    if(tid!=0){ // except first part, require correction
      // add correction back
      for(int i=start_idx; i<end_idx; i++){
        prefix_sum[i]+=correction;
      }
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
