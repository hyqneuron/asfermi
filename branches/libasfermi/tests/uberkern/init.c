#include <assert.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "libasfermi.h"
#include "uberkern.h"
#include "loader.h"

// Generate cubin containing uberkernel with loader
// code and the specified number of free space (in
// instructions).
static struct uberkern_t* uberkern_generate(unsigned int capacity)
{
	int ntokens = sizeof(uberkern) / sizeof(const char*);
	
	// Determine the output size.
	size_t size = 0;
	for (int i = 0, ilines = 0; i < ntokens; i++)
	{
		const char* line = uberkern[i];
			
		// If line starts with '!', then do not count it.
		if (line[0] == '!')
		{
			size += snprintf(NULL, 0, "%s\n", line);
			continue;
		}

		// Account the specified number of NOPs in place of $BUF.
		if (!strcmp(line, "$BUF"))
		{
			for (unsigned int j = 0; j < capacity; j++)
			{
				size += snprintf(NULL, 0, "/*%04x*/ NOP;\n", ilines * 8, line);
				ilines++;
			}
			continue;		
		}
		
		size += snprintf(NULL, 0, "/*%04x*/ %s;\n", ilines * 8, line);
		ilines++;
	}

	// Account additional entry points with all possible
	// reg counts.
	for (unsigned int regcount = 0; regcount < 64; regcount++)
		size += snprintf(NULL, 0, "!Kernel uberkern%d\n!RegCount %d\n"
			"!Param 256 1\nJMP c[0x2][0x8];\n!EndKernel\n", regcount, regcount);

	char* source = (char*)malloc(size + 1);
	char* psource = source;
	
	// Perform the final output.
	for (int i = 0, ilines = 0; i < ntokens; i++)
	{
		const char* line = uberkern[i];
			
		if (line[0] == '!')
		{
			psource += sprintf(psource, "%s\n", line);
			continue;
		}

		// Output the specified number of NOPs in place of $BUF.
		if (!strcmp(line, "$BUF"))
		{
			for (unsigned int j = 0; j < capacity; j++)
			{
				psource += sprintf(psource, "/*%04x*/ NOP;\n", ilines * 8);
				ilines++;
			}
			continue;		
		}
		
		psource += sprintf(psource, "/*%04x*/ %s;\n", ilines * 8, line);
		ilines++;
	}

	// Output additional entry points with all possible
	// reg counts.
	for (unsigned int regcount = 0; regcount < 64; regcount++)
		psource += sprintf(psource, "!Kernel uberkern%d\n!RegCount %d\n"
			"!Param 256 1\nJMP c[0x2][0x8];\n!EndKernel\n", regcount, regcount);

	//printf("%s\n", source);

	char* cubin = asfermi_encode_cubin(source, 20, 0, NULL);
	free(source);
	if (!cubin) return NULL;

	struct uberkern_t* kern = (struct uberkern_t*)malloc(
		sizeof(struct uberkern_t));
	kern->binary = cubin;
	kern->offset = 0;
	kern->capacity = capacity * 8;

	return kern;
}

struct uberkern_t* uberkern_init(unsigned int capacity)
{
	// Generate uberkernel.
	struct uberkern_t* kern = uberkern_generate(capacity);
	if (!kern)
	{
		fprintf(stderr, "Cannot generate uberkernel\n");
		goto failure;
	}

	// Allocate space for uberkernel arguments.
	cudaError_t cudaerr = cudaMalloc((void**)&kern->args, sizeof(struct uberkern_args_t));
	if (cudaerr != cudaSuccess)
	{
		fprintf(stderr, "Cannot allocate space for uberkernel arguments: %d\n",
			cudaerr);
		goto failure;
	}

	// Load binary containing uberkernel to deivce memory.
	CUresult cuerr = cuModuleLoadData(&kern->module, kern->binary);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load uberkernel module: %d\n", cuerr);
		goto failure;
	}
	free(kern->binary);
	kern->binary = NULL;
	
	// Load uberkern loader entry point from module.
	cuerr = cuModuleGetFunction(&kern->loader, kern->module, "uberkern");
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load uberkernel loader entry: %d\n", cuerr);
		goto failure;
	}
	
	// Load uberkernel entry points from module.
	for (int i = 0; i < 64; i++)
	{
		const char* fmt = "uberkern%d";
		int size = snprintf(NULL, 0, fmt, i);
		char* name = (char*)malloc(size + 1);
		sprintf(name, fmt, i);
		cuerr = cuModuleGetFunction(&kern->entry[i], kern->module, name);
		if (cuerr != CUDA_SUCCESS)
		{
			fprintf(stderr, "Cannot load uberkernel entry %s: %d\n", name, cuerr);
			free(name);
			goto failure;
		}
		free(name);
	}

        // Load the uberkernel config structure address constant.
	CUdeviceptr uberkern_config;
	cuerr = cuModuleGetGlobal(&uberkern_config, NULL, kern->module, "uberkern_config");
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load uberkern_config: %d\n", cuerr);
		goto failure;
	}

	// Fill the structure address constant with the address value.
	cuerr = cuMemcpyHtoD(uberkern_config, &kern->args, sizeof(struct uberkern_args_t*));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot fill uberkern_config: %d\n", cuerr);
		goto failure;
	}

        // Load the uberkernel command constant.
	CUdeviceptr uberkern_cmd;
	cuerr = cuModuleGetGlobal(&uberkern_cmd, NULL, kern->module, "uberkern_cmd");
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load uberkern_cmd data: %d\n", cuerr);
		goto failure;
	}

	// Initialize command value with ZERO, so on the next
	// launch uberkern will simply report LEPC and exit.
	cuerr = cuMemsetD8(uberkern_cmd, 0, sizeof(int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot fill uberkern_cmd: %d\n", cuerr);
		goto failure;
	}

	// Launch uberkernel to fill the LEPC.
	// Note we are always sending 256 Bytes, regardless
	// the actual size of arguments.
	char args[256];
	size_t szargs = 256;
	void* config[] =
	{
		CU_LAUNCH_PARAM_BUFFER_POINTER, args,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
		CU_LAUNCH_PARAM_END
	};
	cuerr = cuLaunchKernel(kern->loader,
		1, 1, 1, 1, 1, 1, 0, 0, NULL, config);
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot launch kernel: %d\n", cuerr);
		goto failure;
	}

	// Synchronize kernel.
	cuerr = cuCtxSynchronize();
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot synchronize init kernel: %d\n", cuerr);
		goto failure;
	}

	// Read the LEPC.
	cuerr = cuMemcpyDtoH(&kern->lepc, (CUdeviceptr)kern->args, sizeof(int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot read back the lepc: %d\n", cuerr);
		goto failure;
	}
	printf("uberkern lepc = %p\n", kern->lepc);

	return kern;

failure:
	uberkern_dispose(kern);
	return NULL;
}

