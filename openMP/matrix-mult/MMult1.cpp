// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out
//g++ -std=c++11 -O3 -march=native MMult1.cpp && ./a.out
//g++ -std=c++11 -march=native -o -fopenmp MMult1.cpp && ./a.out
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"

#define BLOCK_SIZE 36 //    optimized block size 36
// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i+p*m];
        double B_pj = b[p+j*k];
        double C_ij = c[i+j*m];
        C_ij = C_ij + A_ip * B_pj;
        c[i+j*m] = C_ij;
      }
    }
  }
}
/*
//Matrix Mult with parrallel looping//
void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  // TODO: See instructions below
    int chunk = 100;
    long j, p, i;
#pragma omp parallel private(j,p,i) shared(a,b,c,m,n,k,chunk)
    {
#pragma omp for schedule (static, chunk)
        for (j = 0; j < n; j++)
        {
            for (p = 0; p < k; p++)
            {
                for (i = 0; i < m; i++)
                {
                    double A_ip = a[i+p*m];
                    double B_pj = b[p+j*k];
                    double C_ij = c[i+j*m];
                    C_ij = C_ij + A_ip * B_pj;
                    c[i+j*m] = C_ij;
                }
            }
        }
    }
}*/
/*

//Matrix mult with blocking//
void MMult1(long m, long n, long k, double *a, double *b, double *c) {
    // TODO: See instructions below
    int chunk = 1;
    double temp;
    //long j,p,i,pp,ii;
    //long jj,kk,i,j,k;
//#pragma omp parallel for shared(a,b,c,m,n,k,chunk) private(j,p,i,pp,ii,temp)
    {
        for ( int i=0; i<n; i+=BLOCK_SIZE ){
            for ( int j=0; j<n; j+=BLOCK_SIZE ){
                for ( int K=0; K<n; K+=BLOCK_SIZE ){
                    for ( int y=i; y<i+BLOCK_SIZE; y++ ){
                        for ( int x=j; x<j+BLOCK_SIZE; x++ ){
                            for ( int z=K; z<K+BLOCK_SIZE; z++ ){
                                double A_ip = a[y+z*m];
                                double B_pj = b[z+x*k];
                                double C_ij = c[y+x*m];
                                //C(y,x) += A(y,z)*B(z,x);
                                C_ij = C_ij + A_ip * B_pj;
                                c[y+x*m] = C_ij;
                            }
                        }
                    }
                }
            }
        }
    }
}*/

//Parellized Matrix mult with blocking//
void MMult1(long m, long n, long k, double *a, double *b, double *c) {
    // TODO: See instructions below
    int chunk = 1000;
    double temp;
    #ifdef _OPENMP
        omp_set_num_threads(4);
    #endif
    #ifdef _OPENMP
    #pragma omp parallel for default(none) shared(a,b,c,chunk) reduction(+:C_ij)
    #endif
    {
        #ifdef _OPENMP
        #pragma omp for schedule(dynamic,chunk) nowait
        #endif
        for ( int i=0; i<n; i+=BLOCK_SIZE ){
            for ( int j=0; j<n; j+=BLOCK_SIZE ){
                for ( int K=0; K<n; K+=BLOCK_SIZE ){
                    for ( int y=i; y<i+BLOCK_SIZE; y++ ){
                        for ( int x=j; x<j+BLOCK_SIZE; x++ ){
                            for ( int z=K; z<K+BLOCK_SIZE; z++ ){
                                double A_ip = a[y+z*m];
                                double B_pj = b[z+x*k];
                                double C_ij = c[y+x*m];
                                //C_ij = C_ij + A_ip * B_pj;
                                C_ij += A_ip * B_pj;
                                c[y+x*m] = C_ij;
                            }
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC = std::max(50/BLOCK_SIZE,1) * BLOCK_SIZE; // multiple of BLOCK_SIZE
    
  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9/(m*n*k)+1;
    double* a = (double*) aligned_malloc(m * k * sizeof(double)); // m x k
    double* b = (double*) aligned_malloc(k * n * sizeof(double)); // k x n
    double* c = (double*) aligned_malloc(m * n * sizeof(double)); // m x n
    double* c_ref = (double*) aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m*k; i++) a[i] = drand48();
    for (long i = 0; i < k*n; i++) b[i] = drand48();
    for (long i = 0; i < m*n; i++) c_ref[i] = 0;
    for (long i = 0; i < m*n; i++) c[i] = 0;

    for (long rep = 0; rep < NREPEATS; rep++) { // Compute reference solution
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();
    double flops = (NREPEATS*m*n*k)/(1e9)/time; // TODO: calculate from m, n, k, NREPEATS, time
    double bandwidth = (sizeof(double)*m*n*k*NREPEATS)/(1e9)/time; // TODO: calculate from m, n, k, NREPEATS, time
    printf("%10d %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m*n; i++) max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.
