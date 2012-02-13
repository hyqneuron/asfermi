/*
 * Copyright (c) 2012 by Dmitry Mikushin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

int sum_host(float* a, float* b, float* c, int n)
{
	int nb = n * sizeof ( float );
	float* aDev = NULL;
	float* bDev = NULL;
	float* cDev = NULL;
	
	int result = 0;	

	void* args[] = { (void*)&aDev, (void*)&bDev, (void*)&cDev };
	CUresult cuerr;

	// Allocate memory on the GPU.
	cudaError_t cudaerr = cudaMalloc((void**)&aDev, nb);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory for aDev: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}
	cudaerr = cudaMalloc((void**)&bDev, nb);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory for bDev: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}
	cudaerr = cudaMalloc((void**)&cDev, nb);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory for cDev: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

	// Copy input data to device memory.
	cudaerr = cudaMemcpy(aDev, a, nb, cudaMemcpyHostToDevice);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from a to aDev: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}
	cudaerr = cudaMemcpy(bDev, b, nb, cudaMemcpyHostToDevice);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from b to bDev: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

	// Load module.
	CUmodule module;
	cuerr = cuModuleLoad(&module, "sum_kernel.cubin");
	assert(cuerr == CUDA_SUCCESS);

	// Load kernel.
	CUfunction kernel;
	cuerr = cuModuleGetFunction(&kernel, module, "kernel");
	assert(cuerr == CUDA_SUCCESS);
	
	// Launch kernel.
	cuerr = cuLaunchKernel(kernel,
		n / BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0,
		0, args, NULL);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel: %d\n", cudaerr);
		result = 1;
		goto finish;
	}

	// Check error status from the launched kernel.
	cudaerr = cudaGetLastError();
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

	// Wait for kernel completion.
	cudaerr = cudaDeviceSynchronize();
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

	// Copy the resulting array back to the host memory.
	cudaerr = cudaMemcpy(c, cDev, nb, cudaMemcpyDeviceToHost);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from cdev to c: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

finish :

	// Release device memory.
	if (aDev) cudaFree(aDev);
	if (bDev) cudaFree(bDev);
	if (cDev) cudaFree(cDev);

	return result;
}

#include <malloc.h>
#include <stdlib.h>

int main ( int argc, char* argv[] )
{
	if (argc != 2)
	{
		printf("Usage: %s <n>\n", argv[0]);
		printf("Where n must be a multiplier of %d\n", BLOCK_SIZE);
		return 0;
	}

	int n = atoi(argv[1]), nb = n * sizeof(float);
	printf("n = %d\n", n);
	if (n <= 0)
	{
		fprintf(stderr, "Invalid n: %d, must be positive\n", n);
		return 1;
	}
	if (n % BLOCK_SIZE)
	{
		fprintf(stderr, "Invalid n: %d, must be a multiplier of %d\n",
			n, BLOCK_SIZE);
		return 1;
	}

	float* a = (float*)malloc(nb);
	float* b = (float*)malloc(nb);
	float* c = (float*)malloc(nb);
	double idrandmax = 1.0 / RAND_MAX;
	for (int i = 0; i < n; i++)
	{
		a[i] = rand() * idrandmax;
		b[i] = rand() * idrandmax;
	}

	int status = sum_host (a, b, c, n);
	if (status) return status;

	int imaxdiff = 0;
	float maxdiff = 0.0;
	for (int i = 0; i < n; i++)
	{
		float diff = c[i] / (a[i] + b[i]);
		if (diff != diff) diff = 0; else diff = 1.0 - diff;
		if (diff > maxdiff)
		{
			maxdiff = diff;
			imaxdiff = i;
		}
	}
	
	printf("Max diff = %f% @ i = %d: %f != %f\n",
		maxdiff * 100, imaxdiff, c[imaxdiff],
		a[imaxdiff] + b[imaxdiff]);
	return 0;
}

