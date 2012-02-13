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

#include "cuda_dyloader.h"

// The Fermi binary for the dummy kernel.
unsigned int kernel[] =
{
	/*0008*/	0x80009de4, 0x28004000,		/* MOV R2, c [0x0] [0x20];	*/
	/*0010*/	0x9000dde4, 0x28004000,		/* MOV R3, c [0x0] [0x24];	*/
	/*0018*/	0x44001de2, 0x18000000,		/* MOV32I R0, 0x11;		*/
	/*0020*/	0x00201c85, 0x94000000,		/* ST.E [R2], R0;		*/
	/*0028*/	0x00001de7, 0x80000000,		/* EXIT;			*/
};

static void usage(const char* filename)
{
	printf("Embedded Fermi dynamic kernels loader\n");
	printf("Usage: %s <capacity>\n", filename);
	printf("\t- where capacity > 0 is the size of free space in kernel\n");
}

int main(int argc, char* argv[])
{
	int result = 0;

	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int capacity = atoi(argv[1]);
	if (capacity <= 0)
	{
		usage(argv[0]);
		return 0;
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
		result = -1;
		goto finish;
	}
	printf("Successfully initialized dynamic loader ...\n");

	// Load kernel function from the binary opcodes.
	CUDYfunction function;
	cuerr = cudyLoadOpcodes(&function,
		loader, (char*)kernel, sizeof(kernel) / 8, 4, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load kernel function: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}

	// Create result buffer.
	char* args = NULL;
	cuerr = cuMemAlloc((CUdeviceptr*)&args, sizeof(void*));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot allocate device memory for kernel args: %d\n",
			cuerr);
		return -1;
	}
	cuerr = cuMemsetD8((CUdeviceptr)args, 0, sizeof(void*));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot initialize device memory for kernel args: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}	

	// Launch kernel function within dynamic loader.
	cuerr = cudyLaunch(function,
		1, 1, 1, 1, 1, 1, 0, &args, 0);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel function: %d\n",
			cuerr);
		result = -1;
		goto finish;
	}
	printf("Launched kernel in uberkernel:\n");
	
	// Synchronize kernel.
	cuerr = cuCtxSynchronize();
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot synchronize target kernel: %d\n", cuerr);
		result = -1;
		goto finish;
	}
	
	// Check the result returned to args.
	void* value = NULL;
	cuerr = cuMemcpyDtoH(&value, (CUdeviceptr)args, sizeof(void*));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot copy result value back to host: %d\n", cuerr);
		result = -1;
		goto finish;
	}
	printf("Done, result = %p\n", value);
	assert(value == (void*)0x11);

finish :
	cudyDispose(loader);

	cuerr = cuMemFree((CUdeviceptr)args);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot free device memory used by kernel args: %d\n",
			cuerr);
		result = -1;
	}	

	// Destroy context.
	cuerr = cuCtxDestroy(context);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot destroy CUDA context: %d\n", cuerr);
		result = -1;
	}

	return result;
}

