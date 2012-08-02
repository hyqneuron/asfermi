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
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_dyloader.h"

// The Fermi binary for the sum_kernel:
// __global__ void sum_kernel ( float * a, float * b, float * c )
// {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     c [idx] = a [idx] + b [idx];
// }
unsigned int kernel[] =
{
	/*0000*/	0x94001c04, 0x2c000000, 	/* S2R R0, SR_CTAid_X;			*/
	/*0008*/	0x84005c04, 0x2c000000,		/* S2R R1, SR_Tid_X;			*/
	/*0010*/	0x2000dca3, 0x20024000,		/* IMAD R3, R0, c [0x0] [0x8], R1;	*/
	/*0018*/	0x1030dca3, 0x5000c000,		/* IMUL R3, R3, 0x4;			*/
	/*0020*/	0x80311c03, 0x48004000,		/* IADD R4, R3, c [0x0] [0x20];		*/
	/*0028*/	0x90015de4, 0x28004000,		/* MOV R5, c [0x0] [0x24];		*/
	/*0030*/	0x00401c85, 0x84000000,		/* LD.E R0, [R4];			*/
	/*0038*/	0xa0311c03, 0x48004000,		/* IADD R4, R3, c [0x0] [0x28];		*/
	/*0040*/	0xb0015de4, 0x28004000, 	/* MOV R5, c [0x0] [0x2c];		*/
	/*0048*/	0x00405c85, 0x84000000,		/* LD.E R1, [R4];			*/
	/*0050*/	0x04001c00, 0x50000000,		/* FADD R0, R0, R1;			*/
	/*0058*/	0xc0311c03, 0x48004000,		/* IADD R4, R3, c [0x0] [0x30];		*/
	/*0060*/	0xd0015de4, 0x28004000, 	/* MOV R5, c [0x0] [0x34];		*/
	/*0068*/	0x00401c85, 0x94000000,		/* ST.E [R4], R0;			*/
	/*0070*/	0x00001de7, 0x80000000,		/* EXIT;				*/
};

// And the following kernel will fail, since it has register footprint
// larger than the predefined amount (uberkernel loader code).
// unsigned int kernel[] =
// {
//	/*0000*/	0x00005de4, 0x28004404,		/* MOV R1, c [0x1] [0x100];			*/
//	/*0008*/	0x94001c04, 0x2c000000,		/* S2R R0, SR_CTAid_X;				*/
//	/*0010*/	0x84009c04, 0x2c000000,		/* S2R R2, SR_Tid_X;				*/
//	/*0018*/	0x10015de2, 0x18000000,		/* MOV32I R5, 0x4;				*/
//	/*0020*/	0x2000dca3, 0x20044000,		/* IMAD R3, R0, c [0x0] [0x8], R2;		*/
//	/*0028*/	0x10311ce3, 0x5000c000,		/* IMUL.HI R4, R3, 0x4;				*/
//	/*0030*/	0x80321ca3, 0x200b8000,		/* IMAD R8.CC, R3, R5, c [0x0] [0x20];		*/
//	/*0038*/	0x90425c43, 0x48004000,		/* IADD.X R9, R4, c [0x0] [0x24];		*/
//	/*0040*/	0xa0329ca3, 0x200b8000,		/* IMAD R10.CC, R3, R5, c [0x0] [0x28];		*/
//	/*0048*/	0x00809c85, 0x84000000,		/* LD.E R2, [R8];				*/
//	/*0050*/	0xb042dc43, 0x48004000,		/* IADD.X R11, R4, c [0x0] [0x2c];		*/
//	/*0058*/	0xc0319ca3, 0x200b8000,		/* IMAD R6.CC, R3, R5, c [0x0] [0x30];		*/
//	/*0060*/	0x00a01c85, 0x84000000,		/* LD.E R0, [R10];				*/
//	/*0068*/	0xd041dc43, 0x48004000,		/* IADD.X R7, R4, c [0x0] [0x34];		*/
//	/*0070*/	0x00201c00, 0x50000000,		/* FADD R0, R2, R0;				*/
//	/*0078*/	0x00601c85, 0x94000000,		/* ST.E [R6], R0;				*/
//	/*0080*/	0x00001de7, 0x80000000,		/* EXIT;					*/
// };

int capacity = 100;

int sum_host(float* a, float* b, float* c, int n)
{
	int nb = n * sizeof ( float );
	float* aDev = NULL;
	float* bDev = NULL;
	float* cDev = NULL;
	
	int result = 0;	

	struct uberkern_t* kern = NULL;

	// Allocate memory on the GPU.
	cudaError_t cuerr = cudaMalloc((void**)&aDev, nb);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory for aDev: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}
	cuerr = cudaMalloc((void**)&bDev, nb);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory for bDev: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}
	cuerr = cudaMalloc((void**)&cDev, nb);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate GPU memory for cDev: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}

	// Copy input data to device memory.
	cuerr = cudaMemcpy(aDev, a, nb, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from a to aDev: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}
	cuerr = cudaMemcpy(bDev, b, nb, cudaMemcpyHostToDevice);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from b to bDev: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}

	// Initialize dynamic loader.
	CUDYloader loader;
	cuerr = cudyInit(&loader, capacity);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize dynamic loader: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}
	printf("Successfully initialized dynamic loader ...\n");

	// Load kernel function from the binary opcodes.
	CUDYfunction function;
	cuerr = cudyLoadOpcodes(&function,
		loader, (char*)kernel, sizeof(kernel) / 8, 6, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load kernel function: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}

	// Launch kernel function within dynamic loader.
	struct { void *aDev, *bDev, *cDev; } args =
		{ .aDev = aDev, .bDev = bDev, .cDev = cDev };
	cuerr = cudyLaunch(function,
		n / BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, &args, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel function: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}
	printf("Launched kernel in uberkernel:\n");

	// Check error status from the launched kernel.
	cuerr = cudaGetLastError();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}

	// Wait for kernel completion.
	cuerr = cudaDeviceSynchronize();
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}

	// Copy the resulting array back to the host memory.
	cuerr = cudaMemcpy(c, cDev, nb, cudaMemcpyDeviceToHost);
	if (cuerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from cdev to c: %s\n",
			cudaGetErrorString(cuerr));
		result = 1;
		goto finish;
	}

finish :

	// Release device memory.
	if (aDev) cudaFree(aDev);
	if (bDev) cudaFree(bDev);
	if (cDev) cudaFree(cDev);

	cudyDispose(loader);
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
	if (status) goto finish;

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

finish:
	free(a);
	free(b);
	free(c);
	return status;
}

