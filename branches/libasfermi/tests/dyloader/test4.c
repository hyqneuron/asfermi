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
#include <malloc.h>
#include <stdio.h>
#include <string.h>

#include "libasfermi.h"
#include "cuda_dyloader.h"

// The Fermi binary for the sum_kernel:
// __global__ void sum_kernel ( float * a1, ..., float * aN, float * b )
// {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     b [idx] = a1 [idx] + ... + aN [idx];
// }
const char* prolog[] =
{
	"!Kernel sum_kernel",
	"!Param 256 1",
	"S2R R0, SR_CTAid_X",
	"S2R R1, SR_Tid_X",
	"IMAD R3, R0, c [0x0] [0x8], R1",
	"IMUL R3, R3, 0x4",
	"MOV R2, RZ",
};
const char* core[] =
{
	"IADD R0, R3, c [0x0] [0x%x]", // index starts with 4 -> 0x20
	"MOV R1, c [0x0] [0x%x]",
	"LD.E R%d, [R0]",
	"FADD R2, R2, R%d",
};
const char* epilog[] =
{
	"IADD R0, R3, c [0x0] [0x%x]", // index starts with 4 -> 0x20
	"MOV R1, c [0x0] [0x%x]",
	"ST.E [R0], R2",
	"EXIT",
	"!EndKernel",
};

int capacity = 100;

int sum_host(CUDYloader loader, float* a, float* b, int n, int narrays)
{
	int nb = n * sizeof ( float );
	float* aDev = NULL;
	float* bDev = NULL;

	void** args = NULL;
	
	int result = 0;	

	// Generate kernel source for the specified number of input arrays.
	// Append "\n" to directive lines and ";\n" to non-directive lines.
	// Insert address and address + 4 in place of %x-s.
	// Insert index in place of %d-s.
	
	// Calculate kernel length.
	size_t szsource = 0;
	for (int i = 0; i < sizeof(prolog) / sizeof(const char*); i++)
	{
		szsource += strlen(prolog[i]) + 1;
		if (prolog[i][0] != '!') szsource++;
	}
	for (int iarray = 0; iarray < narrays; iarray++)
		for (int i = 0, offset = 0; i < sizeof(core) / sizeof(const char*); i++)
		{
			char* insertion = strchr(core[i], '%');
			if (insertion)
			{
				if (insertion[1] == 'x')
				{
					szsource += snprintf(NULL, 0, core[i], (iarray + 4) * 8 + offset);
					offset += 4;
				}
				else if (insertion[1] == 'd')
					szsource += snprintf(NULL, 0, core[i], iarray + 4);
			}
			else
				szsource += strlen(core[i]);
			szsource += 2;
		}
	for (int i = 0, offset = 0; i < sizeof(epilog) / sizeof(const char*); i++)
	{
		char* insertion = strchr(epilog[i], '%');
		if (insertion)
		{
			if (insertion[1] == 'x')
			{
				szsource += snprintf(NULL, 0, epilog[i], (narrays + 4) * 8 + offset);
				offset += 4;
			}
			else if (insertion[1] == 'd')
				szsource += snprintf(NULL, 0, epilog[i], narrays + 4);
		}				
		else
			szsource += strlen(epilog[i]);
		szsource++;
		if (epilog[i][0] != '!') szsource++;
	}
	
	char* source = (char*)malloc(szsource + 1);
	char* psource = source;
	
	// Fill the kernel prolog / core / epilog source.
	for (int i = 0; i < sizeof(prolog) / sizeof(const char*); i++)
	{
		strcpy(psource, prolog[i]);
		psource += strlen(prolog[i]);
		if (prolog[i][0] != '!')
		{
			strcpy(psource, ";");
			psource++;
		}
		strcpy(psource, "\n");
		psource++;
	}
	for (int iarray = 0; iarray < narrays; iarray++)
		for (int i = 0, offset = 0; i < sizeof(core) / sizeof(const char*); i++)
		{
			char* insertion = strchr(core[i], '%');
			if (insertion)
			{
				if (insertion[1] == 'x')
				{
					psource += sprintf(psource, core[i], (iarray + 4) * 8 + offset);
					offset += 4;
				}
				else if (insertion[1] == 'd')
					psource += sprintf(psource, core[i], iarray + 4);				
			}
			else
			{
				strcpy(psource, core[i]);
				psource += strlen(core[i]);
			}
			strcpy(psource, ";\n");
			psource += 2;
		}
	for (int i = 0, offset = 0; i < sizeof(epilog) / sizeof(const char*); i++)
	{
		char* insertion = strchr(epilog[i], '%');
		if (insertion)
		{
			if (insertion[1] == 'x')
			{
				psource += sprintf(psource, epilog[i], (narrays + 4) * 8 + offset);
				offset += 4;
			}
			else if (insertion[1] == 'd')
				psource += sprintf(psource, epilog[i], narrays + 4);
		}
		else
		{
			strcpy(psource, epilog[i]);
			psource += strlen(epilog[i]);
		}
		if (epilog[i][0] != '!')
		{
			strcpy(psource, ";");
			psource++;
		}
		strcpy(psource, "\n");
		psource++;
	}

	//printf("%s", source);
	
	// Compile kernel source to binary opcodes.
	// TODO: size
	size_t szkernel = 0;
	char* kernel = asfermi_encode_cubin(source, 20, 64, &szkernel);
	
	// Allocate memory on the GPU.
	cudaError_t cudaerr = cudaMalloc((void**)&aDev, nb * narrays);
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

	// Copy input data to device memory.
	cudaerr = cudaMemcpy(aDev, a, nb * narrays, cudaMemcpyHostToDevice);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from a to aDev: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

	// Load kernel function from the binary opcodes.
	CUDYfunction function;
	CUresult cuerr = cudyLoadCubin(&function, loader, (char*)kernel, "sum_kernel", 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load kernel function: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}

	// Create arguments list.
	args = (void**)malloc(sizeof(void*) * (narrays + 1));
	for (int i = 0; i < narrays; i++)
		args[i] = aDev + i * n;
	args[narrays] = bDev;
	
	// Launch kernel function within dynamic loader.
	cuerr = cudyLaunch(function,
		n / BLOCK_SIZE, 1, 1, BLOCK_SIZE, 1, 1, 0, args, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel function: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}
	printf("Launched kernel in uberkernel:\n");

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
	cudaerr = cudaMemcpy(b, bDev, nb, cudaMemcpyDeviceToHost);
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot copy data from bDev to b: %s\n",
			cudaGetErrorString(cudaerr));
		result = 1;
		goto finish;
	}

finish :

	// Release device memory.
	if (aDev) cudaFree(aDev);
	if (bDev) cudaFree(bDev);
	if (args) free(args);

	return result;
}

#include <malloc.h>
#include <stdlib.h>

int main ( int argc, char* argv[] )
{
	if (argc != 4)
	{
		printf("Usage: %s <n> <min_narrays> <max_narrays>\n", argv[0]);
		printf("Where n must be a multiplier of %d\n", BLOCK_SIZE);
		printf("Where min_narrays is a minimum number of input arrays in sum\n");
		printf("Where max_narrays is a maximum number of input arrays in sum\n");
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
	
	int min_narrays = atoi(argv[2]);
	if (min_narrays <= 0)
	{
		fprintf(stderr, "The minimum number of input arrays must be positive: %d\n",
			min_narrays);
		return 1; 
	}
	int max_narrays = atoi(argv[3]);
	if (max_narrays < min_narrays)
	{
		fprintf(stderr, "The maximum number of input arrays must be >= than minimum: %d\n",
			max_narrays);
		return 1;
	}
	
	// Initialize driver, select device and create context.
	CUresult cuerr = cuInit(0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize CUDA driver: %d\n", cuerr);
		return -1;
	}
	CUdevice device;
	cuerr = cuDeviceGet(&device, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot get CUDA device #0: %d\n", cuerr);
		return -1;
	}
	CUcontext context;
	cuerr = cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot create CUDA context: %d\n", cuerr);
		return -1;
	}

	// Initialize dynamic loader.
	CUDYloader loader;
	cuerr = cudyInit(&loader, capacity);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize dynamic loader: %d\n",
			cuerr);
		return 1;
	}
	
	for (int narrays = min_narrays; narrays <= max_narrays; narrays++)
	{
		float* a = (float*)malloc(nb * narrays);
		float* b = (float*)malloc(nb);

		double idrandmax = 1.0 / RAND_MAX;
		for (int i = 0; i < n * narrays; i++)
			a[i] = rand() * idrandmax;
		for (int i = 0; i < n; i++)
			b[i] = rand() * idrandmax;

		int status = sum_host (loader, a, b, n, narrays);
		if (status) goto finish;

		int imaxdiff = 0;
		float maxdiff = 0.0;
		for (int i = 0; i < n; i++)
		{
			float sum = 0.0f;
			for (int j = 0; j < narrays; j++)
				sum += a[i + n * j];
			float diff = b[i] / sum;
			if (diff != diff) diff = 0; else diff = 1.0 - diff;
			if (diff > maxdiff)
			{
				maxdiff = diff;
				imaxdiff = i;
			}
		}
		float sum = 0.0f;
		for (int j = 0; j < narrays; j++)
			sum += a[imaxdiff + n * j];
		printf("Max diff = %f% @ i = %d: %f != %f\n",
			maxdiff * 100, imaxdiff, b[imaxdiff], sum);

	finish:
		free(a);
		free(b);
		if (status)
		{
			cudyDispose(loader);
			return status;
		}
	}

	cudyDispose(loader);	
	return 0;
}

