#ifndef BINARY_KERNELS
#define BINARY_KERNELS

#include <stdio.h>
#define BLOCK_SIZE 16

#define DIM_X  16
#define DIM_Y  16

// headers for my kernel implementation
// =============================================================================
// A x B
// size of work for a thread block
#define BLK_M_nn  96
#define BLK_N_nn  96

#define BLK_K  16

// size of thread block for reading A (dev->regs->shmem)
#define DIM_XA  32
#define DIM_YA  8

// size of thread block for reading B (dev->regs->shmem)
#define DIM_XB  8
#define DIM_YB  32

// =============================================================================
#define BLK_M BLK_M_nn
#define BLK_N BLK_N_nn
// =============================================================================

// size of work for a thread
#define THR_M ( BLK_M / DIM_X )
#define THR_N ( BLK_N / DIM_Y )

/******************************************************************************/

#define min(a, b) ((a) < (b) ? (a) : (b))

texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> tex_A;
texture<unsigned int, cudaTextureType1D, cudaReadModeElementType> tex_B;


// CUDA tutorial: http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void gemm(float* A, float* B, float* C, int m, int n, int k) {

    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    float Cvalue = 0.0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        float* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        float* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += As[row][j] * Bs[j][col]; 
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = Cvalue;
}

// 32 single float array ->  32 bits unsigned int
__device__ unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
    
    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>=0);
        rvalue = rvalue | (sign<<i);
    }
    
    return rvalue;
}

__global__ void concatenate_rows_kernel(float *a, unsigned int *b, int size)
{ 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = concatenate(&a[i*32]);
}

__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int m, int n)
{   

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(j<n){
        float * array = new float[32];
        for(int i=0; i<m; i+=32){
            for(int k=0; k<32;k++) array[k] = a[j + n*(i+k)];
            b[j+n*i/32]=concatenate(array); 
        } 
        delete[] array;
    }
}

// 32 bits unsigned int -> 32 single float array
// TODO: the array allocation should not be done here
__device__ float* deconcatenate(unsigned int x)
{
    float * array = new float[32];
    
    for (int i = 0; i < 32; i++)    
    {   
        array[i] = (x & ( 1 << i )) >> i;
    }
    
    return array;
}

__global__ void deconcatenate_rows_kernel(unsigned int *a, float *b, int size)
{ 
    float * array;
    
    for(int i=0; i<size; i+=32)
    {
        array = deconcatenate(a[i/32]);
        for (int k=0;k<32;k++) b[i+k] = array[k];
        delete[] array;
    }
}

// A is shape (m,n), B is shape (n,k) and C is shape (m,k)
__global__ void xnor_gemm(unsigned int* A, unsigned int* B, float* C, int m, int n, int k) {
    
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    float* Csub = &C[BLOCK_SIZE * k * blockRow + BLOCK_SIZE * blockCol];

    // Shared memory used to store Asub and Bsub respectively
    __shared__ unsigned int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ unsigned int Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    // block_size = 16 -> 256 threads, one per Csub element
    unsigned int Cvalue = 0;
    
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int i = 0; i < (n / BLOCK_SIZE); ++i) {
    
        // Get sub-matrix Asub of A
        unsigned int* Asub = &A[BLOCK_SIZE * blockRow * n + BLOCK_SIZE * i];
        
        // Get sub-matrix Bsub of B
        unsigned int* Bsub = &B[BLOCK_SIZE * k * i + BLOCK_SIZE * blockCol];
        
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = Asub[row*n+col];
        Bs[row][col] = Bsub[row*k+col];
    
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        
        // Multiply Asub and Bsub together
        // THIS IS THE MOST INTERESTING PART
        for (int j = 0; j < BLOCK_SIZE; ++j) Cvalue += __popc(As[row][j]^Bs[j][col]);
        
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write Csub to device memory
    // Each thread writes one element
    if(col + blockCol* BLOCK_SIZE< k && row + blockRow* BLOCK_SIZE< m) Csub[row*k+col] = -(2*(float)Cvalue-32*n);
}


__global__ void my_xnor_gemm_kernel(
    int M, int N, int K,
    const unsigned int* __restrict__ A, int LDA,
    const unsigned int* __restrict__ B, int LDB,
    float*       __restrict__ C, int LDC,
    int offsetA, int offsetB )
{

    int idx = threadIdx.x;  // thread's m dimension
    int idy = threadIdx.y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = blockIdx.x;   // block's m dimension
    int bly = blockIdx.y;   // block's n dimension

    __shared__ unsigned int sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ unsigned int sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    unsigned int rC[THR_N][THR_M];
    unsigned int rA[THR_M];
    unsigned int rB[THR_N];

    // Registers for the dev->shmem copy
    unsigned int ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    unsigned int rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

    const unsigned int  *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(K-1) + M) - (blx*BLK_M + idyA*LDA + idxA) - 1;
    const unsigned int *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = 0;

    // Load A dev->shmem
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YA)
            #pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA)
                sA[n+idyA][m+idxA] = offs_dA[min(n*LDA+m, boundA)];

    // Load B dev->shmem
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB)
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                sB[n+idyB][m+idxB] = offs_dB[min(n*LDB+m, boundB)];

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;
        offs_dB += BLK_K;
        boundB  -= BLK_K;

        // Load A dev->regs
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    ra[n][m] = offs_dA[min(n*DIM_YA*LDA + m*DIM_XA, boundA)];

        // Load B dev->regs
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    rb[n][m] = offs_dB[min(n*DIM_YB*LDB + m*DIM_XB, boundB)];

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                        rC[n][m] += __popc(rA[m] ^ rB[n]);
                }
            }
        }

        __syncthreads();

        // Load A regs->shmem
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

        // Load B regs->shmem
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                    rC[n][m] += __popc(rA[m] ^ rB[n]);
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx*BLK_M + m*DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                unsigned int &regC = rC[n][m];
                float &memC = C[offsC];

		memC = -((2 * (float)regC) - (32 * K));
            }
        }
    }
}


__global__ void my_xnortex_gemm_kernel(
    int M, int N, int K,
    const unsigned int* __restrict__ A, int LDA,
    const unsigned int* __restrict__ B, int LDB,
    float*       __restrict__ C, int LDC,
    int offsetA, int offsetB )
{

    int idx = threadIdx.x;  // thread's m dimension
    int idy = threadIdx.y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = blockIdx.x;   // block's m dimension
    int bly = blockIdx.y;   // block's n dimension

    __shared__ unsigned int sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ unsigned int sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    unsigned int rC[THR_N][THR_M];
    unsigned int rA[THR_M];
    unsigned int rB[THR_N];

    // Registers for the dev->shmem copy
    unsigned int ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    unsigned int rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

    int coord_A = offsetA + blx*BLK_M     + idyA*LDA + idxA;
    int coord_B = offsetB + bly*BLK_N*LDB + idyB*LDB + idxB;

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = 0;

    // Load A dev->shmem
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YA)
            #pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA)
		sA[n+idyA][m+idxA] = tex1Dfetch(tex_A, coord_A + n*LDA + m);

    // Load B dev->shmem
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB)
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                sB[n+idyB][m+idxB] = tex1Dfetch(tex_B, coord_B + n*LDB + m);

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
            coord_A += BLK_K*LDA;
            coord_B += BLK_K;

        // Load A dev->regs
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    ra[n][m] = tex1Dfetch(tex_A, coord_A + n*DIM_YA*LDA + m*DIM_XA);

        // Load B dev->regs
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    rb[n][m] = tex1Dfetch(tex_B, coord_B + n*DIM_YB*LDB + m*DIM_XB);

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                        rC[n][m] += __popc(rA[m] ^ rB[n]);
                }
            }
        }

        __syncthreads();

        // Load A regs->shmem
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

        // Load B regs->shmem
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                    rC[n][m] += __popc(rA[m] ^ rB[n]);
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx*BLK_M + m*DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                unsigned int &regC = rC[n][m];
                float &memC = C[offsC];

		memC = -((2 * (float)regC) - (32 * K));
            }
        }
    }
}

__global__ void magma_sgemm_kernel(
    int M, int N, int K,
    const float* __restrict__ A, int LDA,
    const float* __restrict__ B, int LDB,
    float*       __restrict__ C, int LDC,
    int offsetA, int offsetB )
{

    int idx = threadIdx.x;  // thread's m dimension
    int idy = threadIdx.y;  // thread's n dimension

    int idt = DIM_X * idy + idx;    // thread's global number

    int idxA = idt % DIM_XA;    // idx within A
    int idyA = idt / DIM_XA;    // idy within A

    int idxB = idt % DIM_XB;    // idx within B
    int idyB = idt / DIM_XB;    // idy within B

    int blx = blockIdx.x;   // block's m dimension
    int bly = blockIdx.y;   // block's n dimension

    __shared__ float sA[BLK_K][BLK_M+1];      // +1 only required if A is transposed
    __shared__ float sB[BLK_N][BLK_K+1];      // +1 always required

    // Registers for the innermost loop
    float rC[THR_N][THR_M];
    float rA[THR_M];
    float rB[THR_N];

    // Registers for the dev->shmem copy
    float ra[BLK_K/DIM_YA][BLK_M/DIM_XA];
    float rb[BLK_N/DIM_YB][BLK_K/DIM_XB];

    const float  *offs_dA = A + blx*BLK_M     + idyA*LDA + idxA;
    ptrdiff_t boundA = (LDA*(K-1) + M) - (blx*BLK_M + idyA*LDA + idxA) - 1;
    const float *offs_dB = B + bly*BLK_N*LDB + idyB*LDB + idxB;
    ptrdiff_t boundB = (LDB*(N-1) + K) - ( bly*BLK_N*LDB + idyB*LDB + idxB ) -1;

    int m, n, k, kk;

    // Zero C
    #pragma unroll
    for (n = 0; n < THR_N; n++)
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rC[n][m] = 0;

    // Load A dev->shmem
        #pragma unroll
        for (n = 0; n < BLK_K; n += DIM_YA)
            #pragma unroll
            for (m = 0; m < BLK_M; m += DIM_XA)
                sA[n+idyA][m+idxA] = offs_dA[min(n*LDA+m, boundA)];

    // Load B dev->shmem
        #pragma unroll
        for (n = 0; n < BLK_N; n += DIM_YB)
            #pragma unroll
            for (m = 0; m < BLK_K; m += DIM_XB)
                sB[n+idyB][m+idxB] = offs_dB[min(n*LDB+m, boundB)];

    __syncthreads();

    for (kk = 0; kk < K-BLK_K; kk += BLK_K)
    {
        offs_dA += BLK_K*LDA;
        boundA  -= BLK_K*LDA;
        offs_dB += BLK_K;
        boundB  -= BLK_K;

        // Load A dev->regs
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    ra[n][m] = offs_dA[min(n*DIM_YA*LDA + m*DIM_XA, boundA)];

        // Load B dev->regs
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    rb[n][m] = offs_dB[min(n*DIM_YB*LDB + m*DIM_XB, boundB)];

        // Multiply
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load A shmem->regs
            #pragma unroll
            for (m = 0; m < THR_M; m++)
                rA[m] = sA[k][m*DIM_X+idx];

            // Load B shmem->regs
            #pragma unroll
            for (n = 0; n < THR_N; n++)
                rB[n] = sB[n*DIM_Y+idy][k];

            // Compute
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                        rC[n][m] += rA[m] * rB[n];
                }
            }
        }

        __syncthreads();

        // Load A regs->shmem
            #pragma unroll
            for (n = 0; n < BLK_K/DIM_YA; n++)
                #pragma unroll
                for (m = 0; m < BLK_M/DIM_XA; m++)
                    sA[n*DIM_YA+idyA][m*DIM_XA+idxA] = ra[n][m];

        // Load B regs->shmem
            #pragma unroll
            for (n = 0; n < BLK_N/DIM_YB; n++)
                #pragma unroll
                for (m = 0; m < BLK_K/DIM_XB; m++)
                    sB[n*DIM_YB+idyB][m*DIM_XB+idxB] = rb[n][m];

        __syncthreads();
    }

    // Multiply last full (BLK_K) or partial block of
    // columns of op(A) and rows of op(B).
    // It's okay that m,n exceed matrix bounds as all work is in registers
    // or shared memory, and out-of-bounds rC[n][m] will not be saved later.
    kk = K - kk;
    #pragma unroll
    for (k = 0; k < kk; k++)
    {
        // Load A shmem->regs
        #pragma unroll
        for (m = 0; m < THR_M; m++)
            rA[m] = sA[k][m*DIM_X+idx];

        // Load B shmem->regs
        #pragma unroll
        for (n = 0; n < THR_N; n++)
            rB[n] = sB[n*DIM_Y+idy][k];

        // Compute
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                    rC[n][m] += rA[m] * rB[n];
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_dCn = bly*BLK_N + n*DIM_Y + idy;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_dCm = blx*BLK_M + m*DIM_X + idx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                float &regC = rC[n][m];
                float &memC = C[offsC];

		memC = regC;
            }
        }
    }
}

#endif
