#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 16
#define ARRAY_SIZE 1280

int main(int argc, char const *argv[])
{
	// allocate in host
	int *h_a, *h_b, *h_c1, *h_c2;
	cudaMallocHost((void **)&h_a, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);
	cudaMallocHost((void **)&h_b, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);
	cudaMallocHost((void **)&h_c1, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);
	cudaMallocHost((void **)&h_c2, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);

	for (int i = 0; i < ARRAY_SIZE; ++i) {
		for (int j = 0; j < ARRAY_SIZE; ++j) {
			h_a[i * ARRAY_SIZE + j] = rand() % 100;
			h_b[i * ARRAY_SIZE + j] = rand() % 100;
		}
	}
	
	// Allocate in device 
	int *d_a, *d_b, *d_c;
	cudaMalloc((void **)&d_a, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);
	cudaMalloc((void **)&d_b, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);
	cudaMalloc((void **)&d_c, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE);

	// copy from host to device memory
	cudaMemcpy(d_a, h_a, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE, cudaMemcpyHostToDevice);

	int grid_rows = BLOCK_SIZE*BLOCK_SIZE;
	int grid_cols = ARRAY_SIZE / grid_rows;

	dim3 dimGrid(grid_cols, grid_cols,1);
	dim3 dimBlock(grid_rows, grid_rows,1);

	float elapsed_time_gpu;

	//description to calculate time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// start time of GPU
	cudaEventRecord(start, 0);

	matrix_mult_gpu <<<dimGrid, dimBlock >> > (d_a, d_b, d_c, ARRAY_SIZE);
	
	// copy from device to host 
	cudaMemcpy(h_c1, d_c, sizeof(int)*ARRAY_SIZE*ARRAY_SIZE, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	// stop time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// compute time elapse on GPU computing
	cudaEventElapsedTime(&elapsed_time_gpu, start, stop);
	printf("Time elapsed on matrix multiplician %d on GPU: %.1f s.\n\n", ARRAY_SIZE, elapsed_time_gpu);

	// start the CPU version
	clock_t t;
	t = clock();
	matrix_mult_cpu(h_a, h_b, h_c2, ARRAY_SIZE);
	t = clock() - t;
	double elapsed_time = ((double)t) / CLOCKS_PER_SEC;
	printf("Time elapsed on matrix multiplication %d on CPU: %.1f s.\n\n", ARRAY_SIZE, elapsed_time);

	
	
	// free memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c1);
	cudaFreeHost(h_c2);
	return 0;
}


__global__ void matrix_mult_gpu(int *a, int *b, int *c, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < n && row < n)
	{
		for (int i = 0; i < n; i++)
		{
			sum += a[row * n + i] * b[i * n + col];
		}
		c[row * n + col] = sum;
	}
}

void matrix_mult_cpu(int *h_a, int *h_b, int *h_result, int n) 
{
	int i, j, k;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			h_result[i*n + j] = 0;
			for (k = 0; k < n; k++)
			{
				h_result[i*n + j] += h_a[k + i * n] * h_b[k*n + j];
			}
		}
	}
}