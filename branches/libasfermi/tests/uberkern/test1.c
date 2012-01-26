#include <assert.h>
#include <cuda.h>
#include <stdio.h>

#include "uberkern.h"

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

	// Create args.
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
		return -1;
	}	

	// Initialize uberkernel.
	struct uberkern_t* kern = uberkern_init(capacity);
	if (!kern)
	{
		fprintf(stderr, "Cannot initialize uberkernel\n");
		result = -1;
		goto finish;
	}
	printf("Successfully initialized uberkernel ...\n");
	
	// Launch dynamic target kernel in uberkernel.
	void* kernel_args[] = { (void*)&args };
	struct uberkern_entry_t* entry = uberkern_launch(
		kern, NULL, 1, 1, 1, 1, 1, 1, 0,
		kernel_args, (char*)kernel, sizeof(kernel));
	if (!entry)
	{
		fprintf(stderr, "Cannot launch uberkernel\n");
		result = -1;
		goto finish;
	}
	printf("Launched kernel in uberkernel:\n");
	//for (int i = 0; i < sizeof(kernel) / sizeof(int64_t); i++)
	//	printf("0x%016lx\n", kernel[i]);
	
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
	uberkern_dispose(kern);

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

