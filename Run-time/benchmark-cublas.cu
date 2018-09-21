#include <iostream>
#include <cmath>
#include <chrono>
#include <cublas_v2.h>
#include "binary_kernels.cu"

using namespace std;

//to run: nvcc -std=c++11 -O2 -arch=sm_35 --compiler-options "-O2 -Wall -Wextra" -lcublas benchmark-cublas.cu -o bench
int main(int argc, char *argv[]) {

	int M = (argc > 1) ? atoi(argv[1]) : 4096;
	int N = (argc > 2) ? atoi(argv[2]) : 4096;
	int K = (argc > 3) ? atoi(argv[3]) : 4096;
	cout << M << ", " << N << ", " << K << ", ";

	// prepare data
	float *A = (float*)malloc(M * N * sizeof(float));
	float *B = (float*)malloc(N * K * sizeof(float));
	for (int i = 0; i < M * N; i ++) {
		double x = (double)rand() / RAND_MAX;
		A[i] = (x > 0.5) ? 1 : -1;
	}
	for (int i = 0; i < N * K; i ++) {
		double x = (double)rand() / RAND_MAX;
		B[i] = (x > 0.5) ? 1 : -1;
	}


	// copy to cuda
	float *fA, *fB, *fC;
	cudaMalloc(&fA, M * N * sizeof(float));
	cudaMalloc(&fB, N * K * sizeof(float));
	cudaMalloc(&fC, M * K * sizeof(float));
	cudaMemcpy(fA, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(fB, B, N * K * sizeof(float), cudaMemcpyHostToDevice);


	auto test_xnor = [&]() {
		unsigned int *Aconc, *Bconc;
		cudaMalloc(&Aconc, M * N);
		cudaMalloc(&Bconc, N * K);
		cudaMemset(fC, 0, M * K * sizeof(int));

		auto start = chrono::high_resolution_clock::now();
		int block = 64, grid = M * N / (block * 32)  + 1;
		concatenate_rows_kernel<<<grid, block>>>(fA, Aconc, M * N / 32);

		grid = K / block + 1;
		concatenate_cols_kernel<<<grid, block>>>(fB, Bconc, N, K);
		cudaDeviceSynchronize();

		dim3 blockDim(16, 16);
		dim3 gridDim(K / 16 + 1, M / 16 + 1);
		xnor_gemm<<<gridDim, blockDim>>>(Aconc, Bconc, fC, M, N / 32, K);
		cudaDeviceSynchronize();

		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		//cout << "XNOR GEMM kernel time: " << diff.count() << " s\n";
		cout << diff.count() << ", ";

		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(Aconc);
		cudaFree(Bconc);

		return result;
	};
	float* result_xnor = test_xnor();


	/*auto test_my_xnor = [&]() {
		unsigned int *Aconc, *Bconc;
		int M = N;
		int K = N;
		cudaMalloc(&Aconc, M * N * 4 / 32);
		cudaMalloc(&Bconc, N * K * 4 / 32);
		cudaMemset(fC, 0, M * K * sizeof(float));

		auto start = chrono::high_resolution_clock::now();
		int block = 64, grid = M * N / (block * 32)  + 1;
		concatenate_rows_kernel<<<grid, block>>>(fA, Aconc, M * N / 32);

		grid = K / block + 1;
		concatenate_cols_kernel<<<grid, block>>>(fB, Bconc, N, K);
		cudaDeviceSynchronize();

		dim3 blockDim(16, 16);
		int gridSize1 = ceil(static_cast<float>(K) / static_cast<float>(96));
		int gridSize2 = ceil(static_cast<float>(M) / static_cast<float>(96));
		dim3 gridDim(gridSize1, gridSize2);
		my_xnor_gemm_kernel<<<gridDim, blockDim, 0>>>(K, M, N/32, Bconc, K, Aconc, N/32, fC, K, 0, 0);
		cudaDeviceSynchronize();

		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		//cout << "MY XNOR GEMM kernel time: " << diff.count() << " s\n";
		cout << diff.count() << ", ";

		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(Aconc);
		cudaFree(Bconc);

		return result;
        };
	float* result_my_xnor = test_my_xnor();*/


	auto test_my_xnortex = [&]() {
		unsigned int *Aconc, *Bconc;
		cudaMalloc(&Aconc, M * N * 4 / 32);
		cudaMalloc(&Bconc, N * K * 4 / 32);
		cudaMemset(fC, 0, M * K * sizeof(float));

		size_t offsetA = 0;
		size_t offsetB = 0;
		size_t sizeA = (size_t) K * ((N/32) - 1) + K;
		size_t sizeB = (size_t) (N/32) * (M - 1) + (N/32);
		tex_A.normalized = false;
		tex_A.filterMode = cudaFilterModePoint;
		tex_A.addressMode[0] = cudaAddressModeClamp;
		tex_B.normalized = false;
		tex_B.filterMode = cudaFilterModePoint;
		tex_B.addressMode[0] = cudaAddressModeClamp;
		// Bind A and B to texture references
		cudaError_t err;
		err = cudaBindTexture(&offsetA, tex_A, Bconc, sizeA*sizeof(int));
		if ( err != cudaSuccess ) {
			std::exit(1);
		}
		err = cudaBindTexture(&offsetB, tex_B, Aconc, sizeB*sizeof(int));
		if ( err != cudaSuccess ) {
			std::exit(1);
		}
		offsetA = offsetA/sizeof(Bconc[0]);
		offsetB = offsetB/sizeof(Aconc[0]);

		auto start1 = chrono::high_resolution_clock::now();
		int block = 64, grid = M * N / (block * 32)  + 1;
		concatenate_rows_kernel<<<grid, block>>>(fA, Aconc, M * N / 32);

		grid = K / block + 1;
		concatenate_cols_kernel<<<grid, block>>>(fB, Bconc, N, K);
		cudaDeviceSynchronize();

		auto end1 = chrono::high_resolution_clock::now();
		chrono::duration<double> diff1 = end1 - start1;
		cout << diff1.count() << ", ";

		dim3 blockDim(16, 16);
		int gridSize1 = ceil(static_cast<float>(K) / static_cast<float>(96));
		int gridSize2 = ceil(static_cast<float>(M) / static_cast<float>(96));
		dim3 gridDim(gridSize1, gridSize2);
		my_xnortex_gemm_kernel<<<gridDim, blockDim, 0>>>(K, M, N/32, Bconc, K, Aconc, N/32, fC, K, (int)offsetA, (int)offsetB);
		cudaDeviceSynchronize();

		auto end2 = chrono::high_resolution_clock::now();
		chrono::duration<double> diff2 = end2 - end1;
		//cout << "MY XNORTEX GEMM kernel time: " << diff.count() << " s\n";
		cout << diff2.count() << ", ";

		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(Aconc);
		cudaFree(Bconc);

		return result;
        };
	float* result_my_xnortex = test_my_xnortex();

        auto test_magma = [&]() {
		cudaMemset(fC, 0, M * K * sizeof(float));

		size_t offsetA = 0;
		size_t offsetB = 0;

		auto start = chrono::high_resolution_clock::now();

		dim3 blockDim(16, 16);
		int gridSize1 = ceil(static_cast<float>(K) / static_cast<float>(96));
		int gridSize2 = ceil(static_cast<float>(M) / static_cast<float>(96));
		dim3 gridDim(gridSize1, gridSize2);
		magma_sgemm_kernel<<<gridDim, blockDim, 0>>>(K, M, N, fB, K, fA, N, fC, K, (int)offsetA, (int)offsetB);
		cudaDeviceSynchronize();

		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		//cout << "MAGMA SGEMM kernel time: " << diff.count() << " s\n";
		cout << diff.count() << ", ";

		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);

		return result;
        };
	float* result_magma = test_magma();


	auto test_cublas = [&]() {
		cudaMemset(fC, 0, M * K * sizeof(int));
		cublasHandle_t handle;
		cublasCreate(&handle);

		auto start = chrono::high_resolution_clock::now();
		float alpha = 1.0, beta = 0.0;
		// cublas use column-major
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, fB, K, fA, N, &beta, fC, K);
		cudaDeviceSynchronize();
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		//cout << "cublas time: " << diff.count() << " s\n";
		cout << diff.count() << ", ";

		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_cublas = test_cublas();


	auto test_gemm = [&]() {
		cudaMemset(fC, 0, M * K * sizeof(int));
		dim3 blockDim(16, 16);
		dim3 gridDim(K / 16 + 1, M / 16 + 1);
		auto start = chrono::high_resolution_clock::now();
		gemm<<<gridDim, blockDim>>>(fA, fB, fC, M, N, K);
		cudaDeviceSynchronize();
		auto end = chrono::high_resolution_clock::now();
		chrono::duration<double> diff = end - start;
		//cout << "GEMM kernel time: " << diff.count() << " s\n";
		cout << diff.count() << "\n";

		float* result = (float*)malloc(M * K * sizeof(float));
		cudaMemcpy(result, fC, M * K * sizeof(float), cudaMemcpyDeviceToHost);
		return result;
	};
	float* result_gemm = test_gemm();


    /*	
	auto check_result = [&](float* p1, float* p2) {
		for (int i = 0; i < M * K; i ++) {
			float diff = p1[i] - p2[i];
			if (fabs(diff) > 1e-6) {
				printf("%f\n", diff);
				return false;
			}
		}
		return true;
	};
    
	printf("success: %d\n", check_result(result_gemm, result_xnor));
	printf("success: %d\n", check_result(result_gemm, result_cublas));
	printf("success: %d\n", check_result(result_gemm, result_my_xnortex));
	printf("success: %d\n", check_result(result_gemm, result_magma));
	*/

	cudaFree(fA);
	cudaFree(fB);
	cudaFree(fC);
	free(A);
	free(B);
	free(result_gemm);
	free(result_xnor);
	//free(result_my_xnor);
	free(result_my_xnortex);
	free(result_magma);
	free(result_cublas);

	return 0;
}

