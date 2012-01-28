#include <stdio.h>

#include "uberkern.h"

struct uberkern_entry_t* uberkern_launch(
	struct uberkern_t* uberkern, struct uberkern_entry_t* entry,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	size_t szshmem, void* args, char* binary, size_t szbinary)
{
	// Check the dynamic pool has enough free space to
	// incorporate the specified dynamic kernel body.
	if (uberkern->offset + szbinary > uberkern->capacity)
	{
		fprintf(stderr, "Insufficient free space to load the dynamic kernel:\n");
		fprintf(stderr, "%d bytes of %d required bytes are available\n",
			uberkern->capacity - uberkern->offset, szbinary);
		return NULL;
	}
	
	// Load the dynamic kernel code BRA target address.
	CUdeviceptr uberkern_goto;
	CUresult cuerr = cuModuleGetGlobal(&uberkern_goto, NULL, uberkern->module, "uberkern_goto");
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load uberkern_goto: %d\n", cuerr);
		return NULL;
	}

	// Fill the dynamic kernel code BRA target address.
	cuerr = cuMemcpyHtoD(uberkern_goto, &uberkern->offset, sizeof(int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot fill uberkern_goto: %d\n", cuerr);
		return NULL;
	}

	// Load dynamic kernel binary.
	char* binary_dev = NULL;
	cuerr = cuMemAlloc((CUdeviceptr*)&binary_dev, szbinary);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot allocate space for kernel binary on device: %d\n",
			cuerr);
		return NULL;
	}
	cuerr = cuMemcpyHtoD((CUdeviceptr)binary_dev, binary, szbinary);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot copy kernel binary to device: %d\n",
			cuerr);
		return NULL;
	}
	cuerr = cuMemcpyHtoD((CUdeviceptr)&uberkern->args->binary, &binary_dev, sizeof(void*));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot copy kernel binary pointer to device: %d\n",
			cuerr);
		return NULL;
	}
	cuerr = cuMemcpyHtoD((CUdeviceptr)&uberkern->args->szbinary, &szbinary, sizeof(unsigned int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot copy kernel binary size to device: %d\n",
			cuerr);
		return NULL;
	}

	// Note we are always sending 256 Bytes, regardless
	// the actual size of arguments.
	size_t szargs = 256;
	void* config[] =
	{
		CU_LAUNCH_PARAM_BUFFER_POINTER, args,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
		CU_LAUNCH_PARAM_END
	};
	cuerr = cuLaunchKernel(uberkern->function,
		gx, gy, gz, bx, by, bz, szshmem,
		0, NULL, config);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel: %d\n", cuerr);
		return NULL;
	}

	// Increment pool offset by the size of kernel binary.
	uberkern->offset += szbinary;
}

