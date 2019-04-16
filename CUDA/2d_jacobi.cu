#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>
#include <stdlib.h>

void jac2d_serial(double* uk1, double* f, double* u, long N, double h)
{
    //for (long i = 1; i < N*(N-1); i++)
    for (long i = 1; i < N; i++)
    {
        //for (long j = i; j < (i+1)*N-1; j++)
        for (long j = i*N; j < (i+1)*N-1; j++)
        {
        //if(i >= N && i < N*(N-1) && j % N > 0 && j % N < N - 1 )
        if(j % N > 0 && j % N < N - 1 )
        {
            double lft = u[j - 1];    //[i][j-1] col back
            double rght = u[j + 1];   //[i][j+1] col forward
            double top = u[j - N];    //[i-1][j] row up
            double btm = u[j + N];    //[i+1][j] row down
            uk1[j] = (h*h*f[j] + lft + rght + top + btm ) * 0.25;
            //uk1[j] = (h*h*f[j] - lft - rght - top - btm ) * 0.25;
        }
                //uk1[j] = (h*h*f[j] + u[j - 1] + u[j + 1] + u[j - N] + u[j + 1]) * 0.25;
        }
    }
}


void addvec(double* sum_ptr, const double* a, long N)
{
    double sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (long i = 0; i < N*N; i++) sum += a[i];
    *sum_ptr = sum;
}

void Check_CUDA_Error(const char *message){
  cudaError_t error = cudaGetLastError();
  if(error!=cudaSuccess) {
    fprintf(stderr,"ERROR: %s: %s\n", message, cudaGetErrorString(error) );
    exit(-1);
  }
}

#define BLOCK_SIZE 1024

// Warp divergence
//__global__ void reduction_kernel2(double* sum, const double* a, const double* b, long N)

__global__ void reduction_kernel2(double* uk1, double* f, double* u, long N, double h){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  //if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];


if(idx >= N && idx < N*(N-1) && idx % N > 0 && idx % N < N - 1 )
{
double lft = u[idx - 1];    //[i][j-1] col back
double rght = u[idx + 1];   //[i][j+1] col forward
double top = u[idx - N];    //[i-1][j] row up
double btm = u[idx + N];    //[i+1][j] row down

//uk1[idx] = (lft + rght + top + btm + h*h*f[idx]) * 0.25;
smem[threadIdx.x] = (lft + rght + top + btm + h*h*f[idx]) * 0.25;
}
else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) uk1[blockIdx.x] = smem[0] + smem[1];
  }
}

// 2D Jacobi CUDA
__global__ void jacobi_2d(double* uk1, double* f, double* u, long N, double h)
{
//int idx = threadIdx.x;
    //__shared__ double smem[BLOCK_SIZE];
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= N && idx < N*(N-1) && idx % N > 0 && idx % N < N - 1 )
    {
        double lft = u[idx - 1];    //[i][j-1] col back
        double rght = u[idx + 1];   //[i][j+1] col forward
        double top = u[idx - N];    //[i-1][j] row up
        double btm = u[idx + N];    //[i+1][j] row down

        uk1[idx] = (lft + rght + top + btm + h*h*f[idx]) * 0.25;
    }
}

int main() {
    long N = 100;//(1UL<<25);
    double h = 1.0/(N+1.0), sum = 0.0, sum_ref = 0.0;
  double *u; // u
  cudaMallocHost((void**)&u, N * N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) u[i] = 0.0;
double *uhost; // u
cudaMallocHost((void**)&uhost, N * N * sizeof(double));
#pragma omp parallel for schedule(static)
for (long i = 0; i < N*N; i++) uhost[i] = 0.0;

double *uk1;
  cudaMallocHost((void**)&uk1, N * N * sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N*N; i++) uk1[i] = 0.0; // initialize and sets BC's

double *f;
cudaMallocHost((void**)&f, N * N * sizeof(double));
#pragma omp parallel for schedule(static)
for (long i = 0; i < N*N; i++) f[i] = 1.0;

//jac2d_serial( uhost, f, u, N, h);

for (long k = 0; k < 80; k++)
{
    //call to compute Jacobi in serial
    jac2d_serial( uhost, f, u, N, h);
    for (long i = 1; i < N*N; i++) //next iteration
        u[i] = uhost[i];
}
addvec(&sum_ref, uhost, N);

//Reset u for CUDA version
#pragma omp parallel for schedule(static)
for (long i = 0; i < N*N; i++) uk1[i] = 0.0;
#pragma omp parallel for schedule(static)
for (long i = 0; i < N*N; i++) u[i] = 0.0;

  double tt = omp_get_wtime();
  //printf("CPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

double *uk1_d, *u_d, *f_d;//, *y_d;
  cudaMalloc(&uk1_d, N*N*sizeof(double));
cudaMalloc(&u_d, N*N*sizeof(double));
cudaMalloc(&f_d, N*N*sizeof(double));
  //cudaMalloc(&y_d, ((N+BLOCK_SIZE-1)/BLOCK_SIZE)*sizeof(double));

  cudaMemcpyAsync(uk1_d, uk1, N*N*sizeof(double), cudaMemcpyHostToDevice);
//cudaMemcpy(uk1_d, uk1, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
cudaMemcpyAsync(u_d, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
//cudaMemcpy(u_d, u, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
cudaMemcpyAsync(f_d, f, N*N*sizeof(double), cudaMemcpyHostToDevice);
//cudaMemcpy(f_d, f, N*N*sizeof(double), cudaMemcpyHostToDevice);
cudaDeviceSynchronize();
  tt = omp_get_wtime();
//double* ans_d = y_d;
for (long k = 0; k < 80; k++)
{
    //call to compute Jacobi
    jacobi_2d <<< N, BLOCK_SIZE >>> (uk1_d, f_d, u_d, N, h);
    u_d = uk1_d;
}
  cudaMemcpyAsync(uk1, uk1_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  printf("GPU Bandwidth = %f GB/s\n", 1*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
addvec(&sum, uk1, N);
  printf("Error = %f\n", fabs(sum-sum_ref));

printf(" sum_ref: %f\n", sum_ref);
printf(" sum: %f\n", sum);

cudaFree( uk1_d);
cudaFree( u_d);
cudaFree( f_d);
cudaFreeHost( f);
cudaFreeHost( uk1);
cudaFreeHost( uhost);
cudaFreeHost( u);
  return 0;
}



