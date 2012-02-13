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

#include "cuda.h"
#include "cuda_dyloader.h"

__global__ void kernel1(int* lock)
{
	// Wait for unlock.
	while (atomicCAS(lock, 0, 0)) continue;
}

/*extern "C" __global__ void kernel2(int* lock)
{
	// Unlock.
	atomicCAS(lock, 1, 0);
}*/

#include <assert.h>
#include <cuda.h>
#include <stdio.h>

static void usage(const char* filename)
{
	printf("Usage: %s <mode>\n", filename);
	printf("mode = 0: launch kernel1 and then load kernel2 with cuModuleGetFunction (will hang)\n");
	printf("mode = 1: launch kernel1 and then load kernel2 using dyloader (will succeed)\n");
}

int main (int argc, char* argv[])
{
	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int mode = atoi(argv[1]);
	if ((mode < 0) || (mode > 2))
	{
		usage(argv[0]);
		return 0;
	}

	// Initialize lock.
	int* lock = NULL;
	cudaError_t cuerr = cudaMalloc((void**)&lock, sizeof(int));
	assert(cuerr == cudaSuccess);
	int one = 1;
	cuerr = cudaMemcpy(lock, &one, sizeof(int), cudaMemcpyHostToDevice);
	assert(cuerr == cudaSuccess);

	// Dynamic loader initialization is synchronous,
	// no way.
	CUDYloader loader;
	if (mode == 1)
	{
		// Space for dynamically loaded kernels.
		int capacity = 100;
	
		// Initialize dynamic loader.
		CUresult err = cudyInit(&loader, capacity);
		assert(err == CUDA_SUCCESS);
	}

	// Create streams.
	cudaStream_t stream1, stream2;
	cuerr = cudaStreamCreate(&stream1);
	assert(cuerr == cudaSuccess);
	cuerr = cudaStreamCreate(&stream2);
	assert(cuerr == cudaSuccess);

	// Launch first kernel.
	kernel1<<<1, 1, 0, stream1>>>(lock);
	cuerr = cudaGetLastError();
	assert(cuerr == cudaSuccess);

	printf("Submitted kernel1\n");

	if (mode == 0)
	{
		// Load second kernel.
		CUmodule module;
		CUresult err = cuModuleLoad(&module, "kernel2.cubin");
		assert(err == CUDA_SUCCESS);
		CUfunction kernel2;
		err = cuModuleGetFunction(&kernel2, module, "kernel2");

		struct { unsigned int x, y, z; } gridDim, blockDim;
		gridDim.x = 1; gridDim.y = 1; gridDim.z = 1;
		blockDim.x = 1; blockDim.y = 1; blockDim.z = 1;
		size_t szshmem = 0;
		void* kernel2_args[] = { (void*)&lock };
		err = cuLaunchKernel(kernel2,
			gridDim.x, gridDim.y, gridDim.z,
			blockDim.x, blockDim.y, blockDim.z, szshmem,
			stream2, kernel2_args, NULL);
		assert(err == CUDA_SUCCESS);
		
		printf("Sumbitted kernel2\n");

		cuerr = cudaDeviceSynchronize();
		assert(cuerr == cudaSuccess);
	}
	else
	{
		// Load kernel function from the binary opcodes.
		CUDYfunction function;
		CUresult err = cudyLoadCubin(&function,
			loader, "kernel2.cubin", "kernel2", stream2);
		assert(err == CUDA_SUCCESS);

		// Launch kernel function within dynamic loader.
		err = cudyLaunch(function,
			1, 1, 1, 1, 1, 1, 0, &lock, stream2);
		assert(err == CUDA_SUCCESS);
		
		printf("Submitted kernel2\n");

		cuerr = cudaDeviceSynchronize();
		assert(cuerr == cudaSuccess);

		err = cudyDispose(loader);
		assert(err == CUDA_SUCCESS);
	}

	printf("Finished\n");

	cuerr = cudaStreamDestroy(stream1);
	assert(cuerr == cudaSuccess);
	cuerr = cudaStreamDestroy(stream2);
	assert(cuerr == cudaSuccess);

	cuerr = cudaFree(lock);
	assert(cuerr == cudaSuccess);

	return 0;
}

