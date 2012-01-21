#include <assert.h>
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
	"!Kernel uberkern",
	"!Param 8 2",

	"LEPC R0",				// R0 = LEPC
	"MOV R2, c[0x0][0x28]",			// R2 = (int*)&addr
	"MOV R3, c[0x0][0x2c]",			// R3 = (int*)&addr + 1

						// Load the free space starting address from *(int*)(*addr + 1).
	"LDU.E R1, [R2]",			// R1 = *(R2, R3)		<-- 4-byte value of uberkern_args_t.addr
	"IADD R0, R0, R1",			// R0 += R1
	"MOV R1, 0x1",				// R1 = 1			<-- low word compound = 1
	"LD.E.64 R4, [R0]",			// (R4, R5) = *(R0, R1)		<-- code instruction value

						// Load opcode for "NOP" and compare loaded instruction with it.
	"MOV32I R6, -0x00001de4",		// R6 = -0x00001de4
	"IADD R6, R6, R4",			// R6 -= NOP[0]
	"ISETP.NE.AND P0, pt, R6, RZ, pt",	// if (R6 != 0)
	"@P0 BRA #NOP",				// 	goto #NOP
	"MOV32I R6, -0x40000000",		// R6 = -0x40000000
	"IADD R6, R6, R5",			// R6 -= NOP[1]
	"ISETP.NE.AND P0, pt, R6, RZ, pt",	// else if (R6 != 0)
	"@P0 BRA #NOP",				//	goto #NOP

						// OK, if instruction at address goto is pointing to is NOP,
						// then the target kernel is not yet loaded.
						
						// 1) put "goto" opcode specified by addr value in place of #NOP
						// below for further executing.
						
	"LDU.E R1, [R2]",			// R1 = *(R2, R3)		<-- 4-byte value of uberkern_args_t.addr
	"IADD R0, R0, -R1",			// R0 -= R1
	"IADD R0, R0, #NOP",			// R0 += #NOP
	"MOV R1, 0x1",				// R1 = 1			<-- low word compound = 1
	"IADD R2, R2, 8",			// R2 = R2 + 8			<-- address of uberkern_args_t.opcode
	"LD.E.64 R4, [R2]",			// (R4, R5) = *(R2, R3)		<-- 8-byte value of uberkern_args_t.opcode
	"ST.E.64 [R0], R4",			// *(R0, R1) = (R4, R5)
	"IADD R0, R0, -#NOP",			// R0 -= #NOP
	"IADD R2, R2, -8",			// R2 = R2 - 8			<-- address of uberkern_args_t.opcode
	"LDU.E R1, [R2]",			// R1 = *(R2, R3)		<-- 4-byte value of uberkern_args_t.addr
	"IADD R0, R0, R1",
	"MOV R1, 0x1",				// R1 = 1			<-- low word compound = 1


						// 2) load kernel's size and then load each instruction in a loop.
						
	"IADD R2, R2, 16",			// R2 = R2 + 16			<-- address of uberkern_args_t.szbinary
	"LDU.E R6, [R2]",			// R6 = *(R2, R3)
	"IADD R2, R2, 8",			// R2 = R2 + 8			<-- address of uberkern_args_t.binary
	"LDU.E.64 R4, [R2]",			// (R4, R5) = *(R2, R3)
"#L1",	"ISETP.EQ.AND P0, pt, R6, RZ, pt",	// if (R6 == 0)
	"@P0 BRA #NOP",				// 	goto #NOP
						// else
						// {
						//	// Load instructions from args to kernel space
	"LD.E.64 R2, [R4]",			// 	*(R2, R3) = (R4, R5)
	"ST.E.64 [R0], R2",			//	*(R0, R1) = (R2, R3)
	"LD.E.64 R2, [R0]",
	"ST.E.64 [R4], R2",
	"IADD R0, R0, 8",			//	R0 += 8
	"IADD R4, R4, 8",			//	R4 += 8
	"IADD R6, R6, -8",			//	R6 -= 8
	"BRA #L1",				//	goto #L1
						// }
	"MEMBAR.GL",
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
"#NOP",	"NOP",					// goto *(R2, R3)
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

		// Put the specified number of NOPs in place of $BUF.
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

		// Put the specified number of NOPs in place of $BUF.
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

	result = source;

finish :
	free(labels);
	if (!result) return NULL;
	char* cubin = asfermi_encode_cubin(source, 20, 0);
	free(source);
	struct uberkern_t* kern = (struct uberkern_t*)malloc(
		sizeof(struct uberkern_t));
	kern->binary = cubin;
	kern->pool = pool;
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
	CUresult cuerr = cuMemAlloc((CUdeviceptr*)&kern->args, sizeof(struct uberkern_args_t));
	if (cuerr != CUDA_SUCCESS)
	{
		fprintf(stderr, "Cannot allocate space for uberkernel arguments: %d\n",
			cuerr);
		goto failure;
	}

	// Load binary containing uberkernel to deivce memory.
	cuerr = cuModuleLoadData(&kern->module, kern->binary);
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

	return kern;

failure:
	uberkern_dispose(kern);
	return NULL;
}

