#include <stdio.h>

#include "libasfermi.h"
#include "uberkern.h"

struct uberkern_entry_t* uberkern_launch(
	struct uberkern_t* uberkern, struct uberkern_entry_t* entry,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	size_t szshmem, char* args, char* binary, size_t szbinary)
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

	// Generate jump command, create and load opcode.
	unsigned int addr = uberkern->pool + uberkern->offset;
	CUresult cuerr = cuMemcpyHtoD((CUdeviceptr)&uberkern->args->addr,
		&addr, sizeof(unsigned int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load kernel jump address: %d\n",
			cuerr);
		return NULL;
	}
	
	// The goto is exactly before the beginning of the pool, so +8.
	unsigned int offset = uberkern->offset + 8; // uberkern->pool + uberkern->offset;
	const char* fmtcommand = "!Kernel bra\nBRA.U 0x%04x\n!EndKernel\n";
	int szcommand = snprintf(NULL, 0, fmtcommand, offset);
	if (szcommand < 0)
	{
		fprintf(stderr, "Error measuring opcode command length: %d\n",
			szcommand);
		return NULL;
	}
	char command[szcommand];
	sprintf(command, fmtcommand, offset);
	char* opcode = asfermi_encode_opcodes(command, 20);
	if (!opcode)
	{
		fprintf(stderr, "Error generating opcode for command %s\n",
			command);
		return NULL;
	}
	cuerr = cuMemcpyHtoD((CUdeviceptr)&uberkern->args->opcode, opcode, 8);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load kernel jump opcode: %d\n",
			cuerr);
		return NULL;
	}

	printf("goto cmd = BRA.U 0x%04x\n", offset);
	printf("opcode = %p\n", *(void**)opcode);
	free(opcode);

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

	void* kernel_args[] = { (void*)&args };
	cuerr = cuLaunchKernel(uberkern->function,
		gx, gy, gz, bx, by, bz, szshmem,
		0, kernel_args, NULL);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel: %d\n", cuerr);
		return NULL;
	}

	// Increment pool offset by the size of kernel binary.
	uberkern->offset += szbinary;
}

