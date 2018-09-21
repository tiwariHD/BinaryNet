#ifndef CONV_KERNELS
#define CONV_KERNELS

#include "binary_kernels.cu"

#define CUDA_CHECK(condition) \
    /* Code block avoids redefinition of cudaError_t error */ \
        do { \
            cudaError_t error = condition; \
            if (error != cudaSuccess) \
                std::cout << " " << cudaGetErrorString(error); \
        } while (0)

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())

template<typename T>
__global__ void im2col_gpu_kernel(const int n, const T* data_im,
    const int height, const int width, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col,
    T* data_col, T pad_val) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < (n); i += blockDim.x * gridDim.x) {
        
        int w_out = i % width_col;
        int h_i = i / width_col;
        int h_out = h_i % height_col;
        int channel_in = h_i / height_col;
        int channel_out = channel_in * ksize * ksize;
        int h_in = h_out * stride - pad;
        int w_in = w_out * stride - pad;
        T* data_col_ptr = data_col;
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
        const T* data_im_ptr = data_im;
        data_im_ptr += (channel_in * height + h_in) * width + w_in;

        for (int i = 0; i < ksize; ++i) {
            for (int j = 0; j < ksize; ++j) {
                int h = h_in + i;
                int w = w_in + j;
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                    data_im_ptr[i * width + j] : pad_val;
                data_col_ptr += height_col * width_col;
            }
        }
    }
}

void im2col_gpu_float(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col, float pad_val) {

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    int block = 1024;
    int grid = (num_kernels + block - 1) / block;

    im2col_gpu_kernel<float><<<grid, block>>>(num_kernels, data_im, height,
        width, ksize, pad, stride, height_col, width_col, data_col, pad_val);
    
    CUDA_POST_KERNEL_CHECK;
}

void im2col_gpu_int(const unsigned int* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, unsigned int* data_col, unsigned int pad_val) {

	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int num_kernels = channels * height_col * width_col;
    int block = 1024;
    int grid = (num_kernels + block - 1) / block;

    im2col_gpu_kernel<unsigned int><<<grid, block>>>(num_kernels, data_im,
    height, width, ksize, pad, stride, height_col,width_col, data_col, pad_val);
	
    CUDA_POST_KERNEL_CHECK;
}

//run threads <= height * width, blocks = chan/32
//important: channels should b multiple of 32
__global__ void concatenate_input_kernel(float *a, unsigned int *b,
    int height, int width) {

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

__global__ void lifting(const float* data_in, float* data_out,
    int rowSize, int size) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < size; i += blockDim.x * gridDim.x) {

        data_out[i] = data_in[blockIdx.x + (threadIdx.x * rowSize)];
    }
}

void lifting_wrapper(const float* data_in, float* data_out, int row, int column)
{
    lifting<<< column, row  >>>(data_in, data_out, column, row * column);
}

#endif
