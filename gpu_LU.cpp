#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include <iomanip>
#include"cuda_runtime.h"
#include"device_launch_parameters.h"

using namespace std;
const int N = 1024;
const int BLOCK_SIZE = 32;

// 执行高斯消元的核函数
__global__ void gaussianElimination(float* m) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	// 执行高斯消元
	if (row < N && col < N) {
		if (row == col) {
			// 将对角线元素归一化为1
			float pivot = m[row * N + col];
			for (int i = col; i < N; i++) {
				m[row * N + i] /= pivot;
			}
		}
		else {
			// 将非对角线元素消去
			float factor = m[row * N + col] / m[col * N + col];
			for (int i = col; i < N; i++) {
				m[row * N + i] -= factor * m[col * N + i];
			}
		}
	}
}

int main() {
	float* temp = new float[N * N];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			temp[i * N + j] = elm[i][j];
		}
	}

	cudaError_t ret;
	float* gpudata;
	float* result = new float[N * N];
	int size = N * N * sizeof(float);

	ret = cudaMalloc(&gpudata, size);
	if (ret != cudaSuccess) {
		printf("cudaMalloc gpudata failed!\n");
		return 1;
	}

	ret = cudaMemcpy(gpudata, temp, size, cudaMemcpyHostToDevice);
	if (ret != cudaSuccess) {
		printf("cudaMemcpyHostToDevice failed!\n");
		return 1;
	}

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y);

	cudaEvent_t start, stop;
	float elapsedTime = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	gaussianElimination << <dimGrid, dimBlock >> > (gpudata);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("GPU_LU:%f ms\n", elapsedTime);

	cudaError_t cudaStatus2 = cudaGetLastError();
	if (cudaStatus2 != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus2));
		return 1;
	}

	ret = cudaMemcpy(result, gpudata, size, cudaMemcpyDeviceToHost);
	if (ret != cudaSuccess) {
		printf("cudaMemcpyDeviceToHost failed!\n");
		return 1;
	}

	cudaFree(gpudata);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// 打印结果
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)
		{
			cout << setw(10) << result[i * N + j];
		}
		cout << endl;
	}
	delete[] temp;
	delete[] result;

	return 0;
}