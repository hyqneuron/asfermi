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

#ifndef CUDA_DYLOADER_H
#define CUDA_DYLOADER_H

#include <stddef.h>

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
// Additionally, measure the time of the kernel execution if
// the time pointer is not NULL. Note measurement will cause
// synchronization!
CUresult cudyLaunch(CUDYfunction function,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	size_t szshmem, void* args, CUstream stream, float* time);

// Dispose the specified CUDA dynamic loader instance.
CUresult cudyDispose(CUDYloader loader);

#ifdef __cplusplus
}
#endif

#endif // CUDA_DYLOADER_H

