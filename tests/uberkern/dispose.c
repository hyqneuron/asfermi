#include <stdio.h>

#include "uberkern.h"

void uberkern_dispose(struct uberkern_t* uberkern)
{
	if (!uberkern) return;

	if (uberkern->binary) free(uberkern->binary);

	// If args were allocated, then init() was called
	// for uberkern, and stuff was loaded into GPU memory.
	// So, unload it now.
	if (uberkern->args)
	{
		CUresult cuerr = cuMemFree((CUdeviceptr)uberkern->args);
		if (cuerr != CUDA_SUCCESS)
			fprintf(stderr, "Cannot free arguments memory: %d\n", cuerr);
		cuerr = cuModuleUnload(uberkern->module);
		if (cuerr != CUDA_SUCCESS)
			fprintf(stderr, "Cannot unload uberkern module: %d\n", cuerr);
	}
	
	free(uberkern);
}

