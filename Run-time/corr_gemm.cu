// This uses a lot of code from Caffe (http://caffe.berkeleyvision.org/);
// sources are clearly marked. Below we reproduce the original license of
// the Caffe software.
/*
Copyright (c) 2014, The Regents of the University of California (Regents)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#undef _GLIBCXX_ATOMIC_BUILTINS


#define DIM_X  16
#define DIM_Y  16

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

#define CEIL(x) (((x) + 1) / 1)


// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/caffe_common.hpp)
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n);                                       \
       i += blockDim.x * gridDim.x)

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CUDA_NUM_THREADS = 1024;
#else
    const int CUDA_NUM_THREADS = 512;
#endif

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}


// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu)
// Kernels for fast unfold + copy
// CUDA kernel for the case of dilation
__global__ void dilated_im2col_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i * dilation_h;
        int w_im = w_offset + j * dilation_w;
        *data_col_ptr =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * dilation_h * width + j * dilation_w] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

__global__ void im2col_kernel(const int n, const float* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    float* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i ;
        int w_im = w_offset + j ;
        *data_col_ptr =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
           data_im_ptr[i * width + j] : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  if(dilation_h != 1 || dilation_w != 1){
    dilated_im2col_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w,
      dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  }
  else{
    im2col_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  }
}

__global__ void im2col_gpu_int_kernel(const int n, const unsigned int* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    unsigned int* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;
    unsigned int* data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const unsigned int* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i ;
        int w_im = w_offset + j ;
        *data_col_ptr =
          (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
           data_im_ptr[i * width + j] : 0x55555555;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col_gpu_int(const unsigned int* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    unsigned int* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int dil_kernel_h = (kernel_h - 1) * dilation_h + 1;
  int dil_kernel_w = (kernel_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  if(dilation_h != 1 || dilation_w != 1){
     PyErr_Format(PyExc_RuntimeError, "LUL ho gya\n");
    
  }
  else{
    im2col_gpu_int_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, height_col,
      width_col, data_col);
  }
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


__device__ unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;
            
    for (int i = 0; i < 32; i++)
    {
       sign = (array[i]>0);
       rvalue = rvalue | (sign<<i);
    }
                                   
    return rvalue;
}

__global__ void concatenate_input_kernel(float *a, unsigned int *b, int height, int width)
{   
    int size = height * width;
    int out_stride = blockIdx.x * size;
    int in_stride = 32 * out_stride;
    float* array = new float[32];    

    for(int j = threadIdx.x; j < size; j += blockDim.x) {
        for(int k = 0; k < 32; k++)
            array[k] = a[j + in_stride + (k*size)];
        b[j + out_stride] = concatenate(array); 
    }
    delete[] array;
}



// CUDA kernel for the case of dilation
__global__ void dilated_col2im_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO: use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
                                height_col + h_col) * width_col + w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  }
}

__global__ void col2im_kernel(const int n, const float* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    float* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // equivalent implementation, no dilation
    int offset =
      (c_im * kernel_h * kernel_w + h_im * kernel_w + w_im) * height_col * width_col;
    int coeff_h_col = (1 - stride_h * kernel_w * height_col) * width_col;
    int coeff_w_col = (1 - stride_w * height_col * width_col);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

void col2im(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int dilation_h, const int dilation_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im) {
  int dil_patch_h = (patch_h - 1) * dilation_h + 1;
  int dil_patch_w = (patch_w - 1) * dilation_w + 1;
  int height_col = (height + 2 * pad_h - dil_patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - dil_patch_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  if(dilation_h != 1 || dilation_w != 1){
    dilated_col2im_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      dilation_h, dilation_w, pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  }
  else{
    col2im_kernel<<<GET_BLOCKS(num_kernels),
                  CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, patch_h, patch_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im);
  }
}

// Theano op code
// Authors: Arjun Jain, Frederic Bastien, Jan Schluter
// Reference code: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
//   and https://github.com/torch/cunn/blob/master/SpatialConvolutionMM.cu
CudaNdarray* corrMM(CudaNdarray *const bottom,
                    CudaNdarray *const weight,
                    CudaNdarray *const top,
                    const int direction,
                    const int dH = 1,
                    const int dW = 1,
                    const int dilH = 1,
                    const int dilW = 1,
                    const int padH = 0,
                    const int padW = 0)
{
    if (bottom->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires bottom of 4D");
        return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(bottom))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires bottom to be C-contiguous, "
                "but strides are: %d %d %d %d\n",
                CudaNdarray_HOST_STRIDES(bottom)[0],
                CudaNdarray_HOST_STRIDES(bottom)[1],
                CudaNdarray_HOST_STRIDES(bottom)[2],
                CudaNdarray_HOST_STRIDES(bottom)[3]);
        return NULL;
    }

    if (weight->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires weight of 4D");
        return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(weight))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires weight to be C-contiguous, "
                "but strides are: %d %d %d %d\n",
                CudaNdarray_HOST_STRIDES(weight)[0],
                CudaNdarray_HOST_STRIDES(weight)[1],
                CudaNdarray_HOST_STRIDES(weight)[2],
                CudaNdarray_HOST_STRIDES(weight)[3]);
        return NULL;
    }

    if (top->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires top of 4D");
        return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(top))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires top to be C-contiguous, "
                "but strides are: %d %d %d %d\n",
                CudaNdarray_HOST_STRIDES(top)[0],
                CudaNdarray_HOST_STRIDES(top)[1],
                CudaNdarray_HOST_STRIDES(top)[2],
                CudaNdarray_HOST_STRIDES(top)[3]);
        return NULL;
    }

    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth)
    const int batchSize = CudaNdarray_HOST_DIMS(bottom)[0];
    const int nChannels = CudaNdarray_HOST_DIMS(bottom)[1];
    const int bottomHeight = CudaNdarray_HOST_DIMS(bottom)[2];
    const int bottomWidth = CudaNdarray_HOST_DIMS(bottom)[3];
    // weights: (nFilters, nChannels, rows, columns)
    const int nFilters = CudaNdarray_HOST_DIMS(weight)[0];
    const int kH = CudaNdarray_HOST_DIMS(weight)[2];
    const int kW = CudaNdarray_HOST_DIMS(weight)[3];
    if (nChannels != CudaNdarray_HOST_DIMS(weight)[1]) {
        PyErr_SetString(PyExc_ValueError,
                "GpuCorrMM images and kernel must have the same stack size\n");
        return NULL;
    }
    // implicit dilated filter
    const int dil_kH = (kH - 1) * dilH + 1;
    const int dil_kW = (kW - 1) * dilW + 1;
    // top: (batchSize, nFilters, topHeight, topWidth)
    const int topHeightNoDH = (bottomHeight + 2*padH - dil_kH);
    const int topWidthNoDW  = (bottomWidth + 2*padW - dil_kW);
    // the above values might be negative so we need to use Python-like
    // flooring integer division to be compatible with get_conv_output.
    // note: this macro implements Python's // for negative x only
#define _CONV_FLOORDIV_X(x,y) ((x < 0) ? (- ((-x) / y) - (((-x) % y) == 0 ? 0 : 1)) : (x / y))
    const int topHeight = _CONV_FLOORDIV_X(topHeightNoDH, dH) + 1;
    const int topWidth  = _CONV_FLOORDIV_X(topWidthNoDW, dW) + 1;
#undef _CONV_FLOORDIV
    if (batchSize != CudaNdarray_HOST_DIMS(top)[0] ||
            nFilters != CudaNdarray_HOST_DIMS(top)[1] ||
            topHeight != CudaNdarray_HOST_DIMS(top)[2] ||
            topWidth != CudaNdarray_HOST_DIMS(top)[3]) {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM shape inconsistency:\n"
                "  bottom shape: %d %d %d %d\n"
                "  weight shape: %d %d %d %d\n"
                "  top shape: %d %d %d %d (expected %d %d %d %d)\n",
                batchSize, nChannels, bottomHeight, bottomWidth,
                nFilters, nChannels, kH, kW,
                CudaNdarray_HOST_DIMS(top)[0], CudaNdarray_HOST_DIMS(top)[1],
                CudaNdarray_HOST_DIMS(top)[2], CudaNdarray_HOST_DIMS(top)[3],
                batchSize, nFilters, topHeight, topWidth);
        return NULL;
    }

    // Create temporary columns
    int col_dim[2];
    col_dim[0] = nChannels * kW * kH;
    col_dim[1] = topHeight * topWidth;
    CudaNdarray* col = (CudaNdarray*)CudaNdarray_NewDims(2, col_dim);
    if (NULL == col)
    {
        PyErr_Format(PyExc_RuntimeError,
                "GpuCorrMM failed to allocate working memory of %d x %d\n",
                col_dim[0], col_dim[1]);
        return NULL;
    }

    // Define some useful variables
    const int bottom_stride = CudaNdarray_HOST_STRIDES(bottom)[0];
    const int top_stride = CudaNdarray_HOST_STRIDES(top)[0];
    const int K_ = col_dim[0];
    const int N_ = col_dim[1];
    const int M_ = nFilters;
    const float one = 1.0f;
    const float zero = 0.0f;

    CudaNdarray *output;
    if (direction == 0) {  // forward pass
        output = top;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            cudaError_t err = cudaMemset(output->devdata, 0,
                                         CudaNdarray_SIZE(output) * sizeof(real));
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM could not fill the output with zeros: %s",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid correlation: im2col, then gemm
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // First, im2col
            im2col(bottom->devdata + n * bottom_stride, nChannels, bottomHeight,
                   bottomWidth, kH, kW, dilH, dilW,
                   padH, padW, dH, dW, col->devdata);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM encountered a CUDA error in im2col: %s\n"
                             "This could be a known bug in CUDA, please see the "
                             "GpuCorrMM() documentation.\n",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            // Second, gemm
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N_, M_, K_,
                    &one,
                    col->devdata, N_,
                    weight->devdata, K_,
                    &zero,
                    top->devdata + n * top_stride, N_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n"
                        "This could be a known bug in CUDA, please see the "
                        "GpuCorrMM() documentation.\n",
                        cublasGetErrorString(status));
                Py_DECREF(col);
                return NULL;
            }
        }
        /*
        // Original caffe code for comparison
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        // Note that this is for grouped convolution; we can ignore groups here,
        // but the group-related offsets help explain what M_, N_ and K_ are
        int weight_offset = M_ * K_;
        int col_offset = K_ * N_;
        int top_offset = M_ * N_;
        for (int n = 0; n < num_; ++n) {
          // First, im2col
          im2col_gpu(bottom_data + bottom[i]->offset(n), channels_, height_,
              width_, kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
              col_data);
          // Second, innerproduct with groups
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight + weight_offset * g, col_data + col_offset * g,
              (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
            == (see https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L16)
            cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_N,
              N_, M_, K_,
              1.,
              col_data + col_offset * g, N_,
              weight + weight_offset * g, K_,
              0.,
              top_data + (*top)[i]->offset(n) + top_offset * g, N_);
          }
        }
        */
    }
    else if (direction == 1) {  // backprop wrt. weights
        output = weight;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            cudaError_t err = cudaMemset(output->devdata, 0,
                                         CudaNdarray_SIZE(output) * sizeof(real));
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM grad wrt. weights could not fill the output with zeros: %s",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid convolution: im2col, then gemm
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // First, im2col
            im2col(bottom->devdata + n * bottom_stride, nChannels, bottomHeight,
                   bottomWidth, kH, kW, dilH, dilW,
                   padH, padW, dH, dW, col->devdata);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM encountered a CUDA error in im2col: %s\n"
                             "This could be a known bug in CUDA, please see the "
                             "GpuCorrMM() documentation.\n",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            // Second, gemm
            // Note that we accumulate into weight. We do so by setting beta = 0
            // for the first iteration and beta = 1 for subsequent ones. (This
            // is faster than setting weight to all zeros before the loop.)
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    K_, M_, N_,
                    &one,
                    col->devdata, N_,
                    top->devdata + n * top_stride, N_,
                    (n == 0) ? &zero : &one,
                    weight->devdata, K_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n"
                        "This could be a known bug in CUDA, please see the "
                        "GpuCorrMM() documentation.\n",
                        cublasGetErrorString(status));
                Py_DECREF(col);
                return NULL;
            }
        }
        /*
        // Original caffe code for comparison
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        // Note that this is for grouped convolution; we can ignore groups
        for (int n = 0; n < num_; ++n) {
          // Since we saved memory in the forward pass by not storing all col
          // data, we will need to recompute them.
          im2col_gpu(bottom_data + (*bottom)[i]->offset(n), channels_, height_,
                     width_, kernel_h_, kernel_w_, pad_h_, pad_w_,
                     stride_h_, stride_w_, col_data);
          // gradient w.r.t. weight. Note that we will accumulate diffs.
          for (int g = 0; g < group_; ++g) {
            caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                (Dtype)1., top_diff + top[i]->offset(n) + top_offset * g,
                col_data + col_offset * g, (Dtype)1.,
                weight_diff + weight_offset * g);
            == (see https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L16)
            cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, K_, M_, N_,
                1.0,
                col_data + col_offset * g, N_,
                top_diff + top[i]->offset(n) + top_offset * g, N_,
                1.0,
                weight_diff + weight_offset * g, K_);
          }
        }
        */
    }
    else if (direction == 2) {  // backprop wrt. inputs
        output = bottom;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            cudaError_t err = cudaMemset(output->devdata, 0,
                                         CudaNdarray_SIZE(output) * sizeof(real));
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM grad wrt. inputs could not fill the output with zeros: %s",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // full convolution: gemm, then col2im
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // gemm into columns
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    N_, K_, M_,
                    &one,
                    top->devdata + n * top_stride, N_,
                    weight->devdata, K_,
                    &zero,
                    col->devdata, N_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n"
                        "This could be a known bug in CUDA, please see the "
                        "GpuCorrMM() documentation.\n",
                        cublasGetErrorString(status));
                Py_DECREF(col);
                return NULL;
            }
            // col2im back to the data
            col2im(col->devdata, nChannels, bottomHeight, bottomWidth,
                   kH, kW, dilH, dilW, padH, padW,
                   dH, dW, bottom->devdata + n * bottom_stride);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM encountered a CUDA error in col2im: %s\n"
                             "This could be a known bug in CUDA, please see the "
                             "GpuCorrMM() documentation.\n",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
        }
        /*
        // Original caffe code for comparison
        // https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu
        for (int n = 0; n < num_; ++n) {
          // gradient w.r.t. bottom data, if necessary
          if (propagate_down[i]) {
            for (int g = 0; g < group_; ++g) {
              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                  (Dtype)1., weight + weight_offset * g,
                  top_diff + top[i]->offset(n) + top_offset * g,
                  (Dtype)0., col_diff + col_offset * g);
              == (see https://github.com/BVLC/caffe/blob/master/src/caffe/util/math_functions.cu#L16)
              cublasSgemm(CUBLAS_OP_N, CUBLAS_OP_T, N_, K_, M_,
                  1.,
                  top_diff + top[i]->offset(n) + top_offset * g, N_,
                  weight + weight_offset * g, K_,
                  0.,
                  col_diff + col_offset * g, N_);
            }
            // col2im back to the data
            col2im_gpu(col_diff, channels_, height_, width_,
                kernel_h_, kernel_w_, pad_h_, pad_w_, stride_h_, stride_w_,
                bottom_diff + (*bottom)[i]->offset(n));
          }
        }
        */
    }
    // Free temporary columns
    Py_DECREF(col);

    // Note that we don't change the refcount of the output matrix here. Output
    // (re)allocation and refcounting is done in BaseGpuCorrMM.c_code_helper();
    // in here output is just aliased to one of bottom, weights, or top.
    return output;
}

CudaNdarray* corrMMBinary(CudaNdarray *const bottom,
                    CudaNdarray *const weight,
                    CudaNdarray *const top,
                    const int direction,
                    const int dH = 1,
                    const int dW = 1,
                    const int dilH = 1,
                    const int dilW = 1,
                    const int padH = 0,
                    const int padW = 0)
{
    if (bottom->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires bottom of 4D");
        return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(bottom))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires bottom to be C-contiguous, "
                "but strides are: %d %d %d %d\n",
                CudaNdarray_HOST_STRIDES(bottom)[0],
                CudaNdarray_HOST_STRIDES(bottom)[1],
                CudaNdarray_HOST_STRIDES(bottom)[2],
                CudaNdarray_HOST_STRIDES(bottom)[3]);
        return NULL;
    }

    if (weight->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires weight of 4D");
        return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(weight))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires weight to be C-contiguous, "
                "but strides are: %d %d %d %d\n",
                CudaNdarray_HOST_STRIDES(weight)[0],
                CudaNdarray_HOST_STRIDES(weight)[1],
                CudaNdarray_HOST_STRIDES(weight)[2],
                CudaNdarray_HOST_STRIDES(weight)[3]);
        return NULL;
    }

    if (top->nd != 4)
    {
        PyErr_SetString(PyExc_ValueError, "GpuCorrMM requires top of 4D");
        return NULL;
    }
    if (!CudaNdarray_is_c_contiguous(top))
    {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM requires top to be C-contiguous, "
                "but strides are: %d %d %d %d\n",
                CudaNdarray_HOST_STRIDES(top)[0],
                CudaNdarray_HOST_STRIDES(top)[1],
                CudaNdarray_HOST_STRIDES(top)[2],
                CudaNdarray_HOST_STRIDES(top)[3]);
        return NULL;
    }

    // Extract some shape information for later and check shape consistency
    // bottom: (batchSize, nChannels, bottomHeight, bottomWidth)
    const int batchSize = CudaNdarray_HOST_DIMS(bottom)[0];
    const int nChannels = CudaNdarray_HOST_DIMS(bottom)[1];
    const int bottomHeight = CudaNdarray_HOST_DIMS(bottom)[2];
    const int bottomWidth = CudaNdarray_HOST_DIMS(bottom)[3];
    // weights: (nFilters, nChannels, rows, columns)
    const int nFilters = CudaNdarray_HOST_DIMS(weight)[0];
    const int kH = CudaNdarray_HOST_DIMS(weight)[2];
    const int kW = CudaNdarray_HOST_DIMS(weight)[3];
    if (nChannels != CudaNdarray_HOST_DIMS(weight)[1]) {
        PyErr_SetString(PyExc_ValueError,
                "GpuCorrMM images and kernel must have the same stack size\n");
        return NULL;
    }
    // implicit dilated filter
    const int dil_kH = (kH - 1) * dilH + 1;
    const int dil_kW = (kW - 1) * dilW + 1;
    // top: (batchSize, nFilters, topHeight, topWidth)
    const int topHeightNoDH = (bottomHeight + 2*padH - dil_kH);
    const int topWidthNoDW  = (bottomWidth + 2*padW - dil_kW);
    // the above values might be negative so we need to use Python-like
    // flooring integer division to be compatible with get_conv_output.
    // note: this macro implements Python's // for negative x only
#define _CONV_FLOORDIV_X(x,y) ((x < 0) ? (- ((-x) / y) - (((-x) % y) == 0 ? 0 : 1)) : (x / y))
    const int topHeight = _CONV_FLOORDIV_X(topHeightNoDH, dH) + 1;
    const int topWidth  = _CONV_FLOORDIV_X(topWidthNoDW, dW) + 1;
#undef _CONV_FLOORDIV
    if (batchSize != CudaNdarray_HOST_DIMS(top)[0] ||
            nFilters != CudaNdarray_HOST_DIMS(top)[1] ||
            topHeight != CudaNdarray_HOST_DIMS(top)[2] ||
            topWidth != CudaNdarray_HOST_DIMS(top)[3]) {
        PyErr_Format(PyExc_ValueError,
                "GpuCorrMM shape inconsistency:\n"
                "  bottom shape: %d %d %d %d\n"
                "  weight shape: %d %d %d %d\n"
                "  top shape: %d %d %d %d (expected %d %d %d %d)\n",
                batchSize, nChannels, bottomHeight, bottomWidth,
                nFilters, nChannels, kH, kW,
                CudaNdarray_HOST_DIMS(top)[0], CudaNdarray_HOST_DIMS(top)[1],
                CudaNdarray_HOST_DIMS(top)[2], CudaNdarray_HOST_DIMS(top)[3],
                batchSize, nFilters, topHeight, topWidth);
        return NULL;
    }

    // Create temporary columns
    int col_dim[2];
    col_dim[0] = nChannels * kW * kH;
    col_dim[1] = topHeight * topWidth;
    CudaNdarray* col = (CudaNdarray*)CudaNdarray_NewDims(2, col_dim);
    if (NULL == col)
    {
        PyErr_Format(PyExc_RuntimeError,
                "GpuCorrMM failed to allocate working memory of %d x %d\n",
                col_dim[0], col_dim[1]);
        return NULL;
    }


    // Define some useful variables
    const int bottom_stride = CudaNdarray_HOST_STRIDES(bottom)[0];
    const int top_stride = CudaNdarray_HOST_STRIDES(top)[0];
    const int K_ = col_dim[0];
    const int N_ = col_dim[1];
    const int M_ = nFilters;
    const float one = 1.0f;
    const float zero = 0.0f;

    CudaNdarray *output;
    if (direction == 0) {  // forward pass
        output = top;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            cudaError_t err = cudaMemset(output->devdata, 0,
                                         CudaNdarray_SIZE(output) * sizeof(real));
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM could not fill the output with zeros: %s",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
  
        unsigned int* d_input_conc;
        unsigned int* d_kernel_conc;
        unsigned int* d_imColArr;
        d_input_conc = (unsigned int*)device_malloc(nChannels * bottomHeight * bottomWidth * sizeof(unsigned int) / 32);
        d_kernel_conc = (unsigned int*)device_malloc(nFilters * nChannels * kW * kH * sizeof(unsigned int) / 32);
        d_imColArr = (unsigned int*)device_malloc(topHeight * topWidth * nChannels * kW * kH * sizeof(unsigned int) / 32);

        // Concatenate weights
        concatenate_input_kernel<<< (nFilters * nChannels / 32), (kH * kW) >>>(weight->devdata, d_kernel_conc, kH, kW);

        // valid correlation: im2col, then gemm
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // First, concatenate input
            concatenate_input_kernel<<< (nChannels / 32), min(bottomHeight, 256) >>>(
            bottom->devdata + n * bottom_stride, d_input_conc, bottomHeight, bottomWidth);

            // Second, im2col
            im2col_gpu_int(d_input_conc, nChannels / 32, bottomHeight, 
                bottomWidth, kH, kH, dilH, dilW, padH, padH, dH, dW, d_imColArr);
           
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM encountered a CUDA error in im2col: %s\n"
                             "This could be a known bug in CUDA, please see the "
                             "GpuCorrMM() documentation.\n",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }

            dim3 blockDim1(16, 16);
            int gridSize1 = CEIL((float)(N_) / (float)(96));
            int gridSize2 = CEIL((float)(M_) / (float)(96));
            dim3 gridDim1(gridSize1, gridSize2);
            my_xnor_gemm_kernel<<< gridDim1, blockDim1, 0 >>>(N_, M_, (K_)/32,
            d_imColArr, N_, d_kernel_conc, (K_)/32, top->devdata + n * top_stride, N_, 0, 0);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUDA error in im2col: %s\n"
                        "This could be a known bug in CUDA, please see the "
                        "GpuCorrMM() documentation.\n",
                        cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
        }
        
        device_free(d_input_conc);
        d_input_conc = NULL;
        device_free(d_kernel_conc);
        d_kernel_conc = NULL;;
        device_free(d_imColArr);
        d_imColArr = NULL;;
    }
    else if (direction == 1) {  // backprop wrt. weights
        output = weight;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            cudaError_t err = cudaMemset(output->devdata, 0,
                                         CudaNdarray_SIZE(output) * sizeof(real));
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM grad wrt. weights could not fill the output with zeros: %s",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // valid convolution: im2col, then gemm
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // First, im2col
            im2col(bottom->devdata + n * bottom_stride, nChannels, bottomHeight,
                   bottomWidth, kH, kW, dilH, dilW,
                   padH, padW, dH, dW, col->devdata);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM encountered a CUDA error in im2col: %s\n"
                             "This could be a known bug in CUDA, please see the "
                             "GpuCorrMM() documentation.\n",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            // Second, gemm
            // Note that we accumulate into weight. We do so by setting beta = 0
            // for the first iteration and beta = 1 for subsequent ones. (This
            // is faster than setting weight to all zeros before the loop.)
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    K_, M_, N_,
                    &one,
                    col->devdata, N_,
                    top->devdata + n * top_stride, N_,
                    (n == 0) ? &zero : &one,
                    weight->devdata, K_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n"
                        "This could be a known bug in CUDA, please see the "
                        "GpuCorrMM() documentation.\n",
                        cublasGetErrorString(status));
                Py_DECREF(col);
                return NULL;
            }
        }
    }
    else if (direction == 2) {  // backprop wrt. inputs
        output = bottom;
        if (batchSize == 0 || nChannels == 0 || nFilters == 0) {
            cudaError_t err = cudaMemset(output->devdata, 0,
                                         CudaNdarray_SIZE(output) * sizeof(real));
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM grad wrt. inputs could not fill the output with zeros: %s",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
            Py_DECREF(col);
            return output;
        }
        // full convolution: gemm, then col2im
        // Iterate over batch
        for (int n = 0; n < batchSize; n++) {
            // gemm into columns
            cublasStatus_t status = cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    N_, K_, M_,
                    &one,
                    top->devdata + n * top_stride, N_,
                    weight->devdata, K_,
                    &zero,
                    col->devdata, N_);
            if (status != CUBLAS_STATUS_SUCCESS) {
                PyErr_Format(PyExc_RuntimeError,
                        "GpuCorrMM encountered a CUBLAS error: %s\n"
                        "This could be a known bug in CUDA, please see the "
                        "GpuCorrMM() documentation.\n",
                        cublasGetErrorString(status));
                Py_DECREF(col);
                return NULL;
            }
            // col2im back to the data
            col2im(col->devdata, nChannels, bottomHeight, bottomWidth,
                   kH, kW, dilH, dilW, padH, padW,
                   dH, dW, bottom->devdata + n * bottom_stride);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                PyErr_Format(PyExc_RuntimeError,
                             "GpuCorrMM encountered a CUDA error in col2im: %s\n"
                             "This could be a known bug in CUDA, please see the "
                             "GpuCorrMM() documentation.\n",
                             cudaGetErrorString(err));
                Py_DECREF(col);
                return NULL;
            }
        }
    }
    // Free temporary columns
    Py_DECREF(col);

    return output;
}

CudaNdarray* corrMMWrapper(CudaNdarray *const bottom,
                    CudaNdarray *const weight,
                    CudaNdarray *const top,
                    int direction,
                    int dH, int dW,
                    int dilH, int dilW,
                    int padH, int padW,
                    const int callBinary = 0)
{
    if (callBinary == 1) {
        return corrMMBinary(bottom, weight, top, direction, dH, dW, dilH, dilW, padH, padW);
    } else {
        return corrMM(bottom, weight, top, direction, dH, dW, dilH, dilW, padH, padW);
    }
}

