#ifndef CUDA_DYLOADER_H
#define CUDA_DYLOADER_H

#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct CUDYloader_t* CUDYloader;
typedef struct CUDYfunction_t* CUDYfunction;

// Initialize a new instance of CUDA dynamic loader with the
// specified capacity (in 8-byte instructions) in GPU memory.
CUresult cudyInit(CUDYloader* loader, int capacity);

// Load kernel function with the specified name from cubin file
// or memory buffer into dynamic loader context.
CUresult cudyLoadCubin(CUDYfunction* function,
	CUDYloader loader, char* cubin, const char* name,
	CUstream stream);

// Load kernel function from the specified assembly opcodes
// into dynamic loader context.
CUresult cudyLoadOpcodes(CUDYfunction* function,
	CUDYloader loader, char* opcodes, size_t nopcodes,
	int regcount, CUstream stream);

// Launch kernel function through the dynamic loader.
CUresult cudyLaunch(CUDYfunction function,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	size_t szshmem, void* args, CUstream stream);

// Dispose the specified CUDA dynamic loader instance.
CUresult cudyDispose(CUDYloader loader);

#ifdef __cplusplus
}
#endif

#endif // CUDA_DYLOADER_H

