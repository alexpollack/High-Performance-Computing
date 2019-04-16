# High-Performance-Computing
This is a set of codes exploring various high-performance computing tools, including parallel and distributed computing, such as the CUDA and OpenMP.
## CUDA
The file CUDA contains a few programs that compare serial/parallel code run on CPU to similar functions processed with CUDA. These files are meant to be run on NVIDIA GPU as they are CUDA based. 

#### dotprod.cu
Compute vector inner products, comparing parallel OpenMP CPU to CUDA GPU 

#### mvmult.cu
Compute matrix vector products, comparing parallel OpenMP CPU to CUDA GPU 

#### 2d_jacobi.cu
Compute the 2D Jacobi method. The actual error is not produced, but rather ran for a large number of iterations and then the resultant solutions are summed and the error is the difference in the sum of the standard CPU version and the implemented GPU version.

## OpenMP
The file openMP contains code of different algorithms coded in parallel using OpenMP.

#### matrix-mult
Within the file matrix-mult, the code MMult1.cpp contains serial, parallel, and parallel with blocking versions of large matrix multiplication using OpenMP.

## Other-HPC
### Intrinsic Functions and Vectorized Functions
#### fast-sin
Within the file fast-sin, fast-sin.cpp has functions to compute the Taylor series expansion for sin(x). This includes writing it has an intrinsic function (both set for options AVX or SSE2 dependent on the machine), as well as a vectorized version. 
