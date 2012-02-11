#include <assert.h>
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "libasfermi.h"
#include "uberkern.h"

// Loader source code template.
// uberkern (int* args, void** addr)
static const char* uberkern[] =
{
	"!Machine 64",

	"!Constant2 0x10",			// Reserve 16 bytes of constant memory to store:
	"!Constant long 0x0 uberkern_config",	// the address of uberkern config structure
	"!EndConstant",
	"!Constant int 0x8 uberkern_cmd",	// select command:
						// 0 - load effective PC of uberkern
						// 1 - load dynamic kernel source code
						// other value - execute dynamic kernel
	"!EndConstant",
	"!Constant int 0xc uberkern_goto",	// the dynamic kernel relative offset in uberkern
	"!EndConstant",

	"!Kernel uberkern",
	"!Param 256 1",

	"LEPC R0",				// R0 = LEPC

	"MOV R2, c[0x2][0x8]",			// Check if the uberkern_cmd contains 0.
	"ISETP.NE.AND P0, pt, R2, RZ, pt",
	"@P0 BRA #LD",				// If not - go to #LD
	"MOV R2, c[0x2][0x0]",
	"MOV R3, c[0x2][0x4]",
	"ST.E [R2], R0",			// If yes, write LEPC to uberkern_config and exit.
	"EXIT",

"#LD",	"MOV R1, 0x1",				// Check if the uberkern_cmd contains 1.
	"ISETP.NE.AND P0, pt, R2, R1, pt",
	"@P0 BRA #BRA",				// If not - go to #BRA

						// If yes, write dynamic kernel code and exit.

						// Load the dynamic kernel starting address.

"#GO",	"MOV R1, c[0x2][0xc]",			// R1 = c[0x2][0x12]		<-- 4-byte value of goto offset
	"IADD R1, R1, #FRE",			// R1 += #FRE
	"IADD R0, R0, R1",			// R0 += R1
	"MOV R1, 0x1",				// R1 = 1			<-- low word compound = 1

						// Load kernel's size and then load each instruction in a loop.

	"MOV R2, c[0x2][0x0]",
	"MOV R3, c[0x2][0x4]",
	"LD.E R6, [R2]",			// R6 = *(R2, R3)
	"IADD R2, R2, 8",			// R2 = R2 + 8			<-- address of uberkern_args_t.binary
	"LD.E.64 R4, [R2]",			// (R4, R5) = *(R2, R3)
"#L1",	"ISETP.EQ.AND P0, pt, R6, RZ, pt",	// if (R6 == 0)
	"@P0 EXIT",				// 	exit;
						// else
						// {
						//	// Load instructions from args to kernel space
	"LD.E.64 R2, [R4]",			// 	*(R2, R3) = (R4, R5)
	"ST.E.64 [R0], R2",			//	*(R0, R1) = (R2, R3)
	"IADD R0, R0, 8",			//	R0 += 8
	"IADD R4, R4, 8",			//	R4 += 8
	"IADD R6, R6, -8",			//	R6 -= 8
	"BRA #L1",				//	goto #L1
						// }
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
	"NOP",
"#BRA",	"BRA c[0x2][0xc]",			// goto dynamic kernel offset
"#FRE",	"$BUF",					// $BUF more NOPs here as free space for code insertions
	"!EndKernel"
};

// Generate cubin containing uberkernel with loader
// code and the specified number of free space (in
// instructions).
static struct uberkern_t* uberkern_generate(unsigned int capacity)
{
	char* result = NULL;
	unsigned int pool;

	int ntokens = sizeof(uberkern) / sizeof(const char*);
	
	// Record the number of labels.
	int nlabels = 0;
	for (int i = 0; i < ntokens; i++)
	{
		const char* line = uberkern[i];
		if (line[0] == '#') nlabels++;
	}

	// Record labels themselves and their positions.
	struct label_t
	{
		// The name of label.
		const char* name;
	
		// The address of instruction label is pointing to.
		int addr;
	};
	struct label_t* labels = (struct label_t*)malloc(
		nlabels * sizeof(struct label_t));		
	for (int i = 0, ilabel = 0, nskip = 0; i < ntokens; i++)
	{
		const char* line = uberkern[i];
		
		if (line[0] == '!')
		{
			nskip++;
			continue;
		}

		if (line[0] == '#')
		{
			if (strlen(line) < 2)
			{
				fprintf(stderr, "Invalid label: \"%s\"\n", line);
				goto finish;
			}
			labels[ilabel].name = line;
			labels[ilabel].addr = (i - nskip - ilabel) * 8;

			// Record where #FRE label points to: it would
			// be the starting address for dynamically loaded
			// kernels.
			if (!strcmp(line, "#FRE"))
				pool = labels[ilabel].addr;

			ilabel++;
		}
	}
	
	// Determine the output size.
	size_t size = 0;
	for (int i = 0, ilines = 0; i < ntokens; i++)
	{
		const char* line = uberkern[i];
		int szline = strlen(line);
			
		// Skip lines-labels.
		if (line[0] == '#') continue;

		// If line starts with '!', then just print it, do not count.
		if (line[0] == '!')
		{
			size += snprintf(NULL, 0, "%s\n", line);
			continue;
		}

		// Account the specified number of NOPs in place of $BUF.
		if (!strncmp(line, "$BUF", 5))
		{
			for (unsigned int j = 0; j < capacity; j++)
			{
				size += snprintf(NULL, 0, "/*%04x*/ NOP;\n", ilines * 8, line);
				ilines++;
			}
			continue;		
		}
		
		// Search for single label in each line.
		struct label_t* maxlabel = NULL;
		for (int j = 0; j < szline; j++)
		{
			if (line[j] != '#') continue;
			
			// Find the longest matching label.
			for (int ilabel = 0; ilabel < nlabels; ilabel++)
			{
				struct label_t* label = labels + ilabel;
				if (strcmp(line + j, label->name)) continue;
				
				if (!maxlabel || (strlen(label->name) > strlen(maxlabel->name)))
					maxlabel = label;
			}
			
			if (!maxlabel)
			{
				fprintf(stderr, "Used label not found: %s\n", line);
				goto finish;
			}
			
			// Build new line.
			int szlabel = strlen(maxlabel->name);
			int sznewline = szline - szlabel + 6;
			char* newline = (char*)malloc(sznewline + 1);
			memcpy(newline, line, j);
			sprintf(newline + j, "0x%04x", maxlabel->addr);
			memcpy(newline + j + 6, line + j + szlabel,
				szline - j - szlabel);
			size += snprintf(NULL, 0, "/*%04x*/ %s;\n", ilines * 8, newline);
			ilines++;
			free(newline);
			break;
		}

		if (!maxlabel)
		{
			size += snprintf(NULL, 0, "/*%04x*/ %s;\n", ilines * 8, line);
			ilines++;
		}
	}

	// Account additional entry points with all possible
	// reg counts.
	for (unsigned int regcount = 0; regcount < 64; regcount++)
		size += snprintf(NULL, 0, "!Kernel uberkern%d\n!RegCount %d\n"
			"!Param 256 1\nJMP c[0x2][0x8]\n!EndKernel\n", regcount, regcount);

	char* source = (char*)malloc(size + 1);
	char* psource = source;
	
	// Perform the final output.
	for (int i = 0, ilines = 0; i < ntokens; i++)
	{
		const char* line = uberkern[i];
		int szline = strlen(line);
			
		// Skip lines-labels.
		if (line[0] == '#') continue;

		// If line starts with '!', then just print it, do not count.
		if (line[0] == '!')
		{
			psource += sprintf(psource, "%s\n", line);
			continue;
		}

		// Output the specified number of NOPs in place of $BUF.
		if (!strncmp(line, "$BUF", 5))
		{
			int sznop = sprintf(psource, "/*%04x*/ NOP;\n", ilines * 8, line);
			psource -= (ptrdiff_t)source;
			source = (char*)realloc(source, size + sznop * capacity);
			psource += (ptrdiff_t)source;
			for (int j = 1; j < capacity; j++)
			{
				psource += sprintf(psource, "/*%04x*/ NOP;\n", ilines * 8, line);
				ilines++;
			}
			continue;		
		}
		
		// Search for single label in each line.
		struct label_t* maxlabel = NULL;
		for (int j = 0; j < szline; j++)
		{
			if (line[j] != '#') continue;
			
			// Find the longest matching label.
			for (int ilabel = 0; ilabel < nlabels; ilabel++)
			{
				struct label_t* label = labels + ilabel;
				if (strcmp(line + j, label->name)) continue;
				
				if (!maxlabel || (strlen(label->name) > strlen(maxlabel->name)))
					maxlabel = label;
			}
						
			// Build new line.
			int szlabel = strlen(maxlabel->name);
			int sznewline = szline - szlabel + 6;
			char* newline = (char*)malloc(sznewline + 1);
			memcpy(newline, line, j);
			sprintf(newline + j, "0x%04x", maxlabel->addr);
			memcpy(newline + j + 6, line + j + szlabel,
				szline - j - szlabel);
			psource += sprintf(psource, "/*%04x*/ %s;\n", ilines * 8, newline);
			ilines++;
			free(newline);
			break;
		}

		if (!maxlabel)
		{
			psource += sprintf(psource, "/*%04x*/ %s;\n", ilines * 8, line);
			ilines++;
		}
	}

	// Output additional entry points with all possible
	// reg counts.
	for (unsigned int regcount = 0; regcount < 64; regcount++)
		size += snprintf(NULL, 0, "!Kernel uberkern%d\n!RegCount %d\n"
			"!Param 256 1\nJMP c[0x2][0x8]\n!EndKernel\n", regcount, regcount);

	result = source;

finish :
	free(labels);
	if (!result) return NULL;
	char* cubin = asfermi_encode_cubin(source, 20, 0, NULL);
	free(source);
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
	
	// Load uberkernel entry point from module.
	cuerr = cuModuleGetFunction(&kern->function, kern->module, "uberkern");
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot load uberkernel function: %d\n", cuerr);
		goto failure;
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
	cuerr = cuLaunchKernel(kern->function,
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
		fprintf(stderr, "Cannot synchronize target kernel: %d\n", cuerr);
		goto failure;
	}

	// Read the LEPC.
	unsigned int lepc;
	cuerr = cuMemcpyDtoH(&lepc, (CUdeviceptr)kern->args, sizeof(int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot read back the lepc: %d\n", cuerr);
		goto failure;
	}
	printf("uberkern lepc = %p\n", lepc);

	// Again, initialize the LEPC, this time with the actual value.
	cuerr = cuMemcpyHtoD(uberkern_cmd, &lepc, sizeof(int));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot fill uberkern_cmd: %d\n", cuerr);
		goto failure;
	}

	return kern;

failure:
	uberkern_dispose(kern);
	return NULL;
}

