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

int main(int argc, char** argv) 
{
	// Create device context.
	CUdevice device;
	CUcontext context;
	CUresult cuerr = cuInit(0);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuDeviceGet(&device, 0);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device);
	assert(cuerr == CUDA_SUCCESS);

	// Create an output value buffer.
	int* value;
	cuerr = cuMemAlloc((CUdeviceptr*)&value, sizeof(int));
	assert(cuerr == CUDA_SUCCESS);

	// Load module.
	CUmodule module;
	cuerr = cuModuleLoad(&module, "constant3.cubin");
	assert(cuerr == CUDA_SUCCESS);

	// Load the unnamed constant (that is given a name, anyway).
	CUdeviceptr unnamed;
	size_t szunnamed = 0;
	cuerr = cuModuleGetGlobal(&unnamed, &szunnamed, module,
		"unnamedConst2_0");
	assert(cuerr == CUDA_SUCCESS);
	printf("unnamedConst2_0 addr = %p size = %zu\n", (void*)unnamed, szunnamed);

	// Load the named constant.
	CUdeviceptr named;
	size_t sznamed = 0;
	cuerr = cuModuleGetGlobal(&named, &sznamed, module,
		"named");
	assert(cuerr == CUDA_SUCCESS);
	printf("named addr = %p, size = %zu\n", (void*)named, sznamed);

	// Load kernel.
	CUfunction kernel;
	cuerr = cuModuleGetFunction(&kernel, module, "kernel");
	assert(cuerr == CUDA_SUCCESS);
	
	// Configure kernel launch with output buffer parameter.
	cuerr = cuParamSetSize(kernel, 8);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuParamSetv(kernel, 0, &value, 8);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuFuncSetBlockShape(kernel, 1, 1, 1);
	assert(cuerr == CUDA_SUCCESS);

	// Launch kernel.
	cuerr = cuLaunch(kernel);
	assert(cuerr == CUDA_SUCCESS);

	// Wait for kernel completion.
	cuerr = cuCtxSynchronize();
	assert(cuerr == CUDA_SUCCESS);

	// Get result from device memory.
	int cpu_value;
	cuerr = cuMemcpyDtoH(&cpu_value, (CUdeviceptr)value, sizeof(int));
	assert(cuerr == CUDA_SUCCESS);
	printf("initial result = %d\n", cpu_value);

	// Change the value in contant memory and run
	// kernel again to copy it to the output buffer.
	cpu_value = 10;
	cuerr = cuMemcpyHtoD(unnamed, &cpu_value, sizeof(int));
	assert(cuerr == CUDA_SUCCESS);
	cpu_value = 0;

	// Configure kernel launch with output buffer parameter.
	cuerr = cuParamSetSize(kernel, 8);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuParamSetv(kernel, 0, &value, 8);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuFuncSetBlockShape(kernel, 1, 1, 1);
	assert(cuerr == CUDA_SUCCESS);

	// Launch kernel.
	cuerr = cuLaunch(kernel);
	assert(cuerr == CUDA_SUCCESS);

	// Wait for kernel completion.
	cuerr = cuCtxSynchronize();
	assert(cuerr == CUDA_SUCCESS);

	// Get result from device memory.
	cuerr = cuMemcpyDtoH(&cpu_value, (CUdeviceptr)value, sizeof(int));
	assert(cuerr == CUDA_SUCCESS);
	printf("changed result = %d\n", cpu_value);

	// Free output buffer.
	cuerr = cuMemFree((CUdeviceptr)value);
	assert(cuerr == CUDA_SUCCESS);

	// Unload module and destroy context.
        cuerr = cuModuleUnload(module);
	assert(cuerr == CUDA_SUCCESS);
	cuerr = cuCtxDestroy(context);
	assert(cuerr == CUDA_SUCCESS);

        return 0;
}

