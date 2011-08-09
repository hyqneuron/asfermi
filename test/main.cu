#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

extern "C"
{
	__global__ void testKernel(int* addr, unsigned short param1, char param2)
	{
		addr[0] = param1 + param2;
	}
}

char* muGetErrorString(CUresult result);

void muEC(int position) //checks and outputs error position and error string
{
	cudaError_t errcode = cudaGetLastError();
	if(errcode==cudaSuccess)
	{
		printf("No error at position %i\n", position);
		return;
	}
	printf("Error position: %i\nCode:%s\n", position, cudaGetErrorString(errcode));
}

void muRC(int position, CUresult result)
{
	if(result==0)
		printf("Success at %i\n", position);
	else
		printf("Error at %i:%s\n", position, muGetErrorString(result));
}

char* muGetErrorString(CUresult result)
{
	switch(result)
	{
	case 0:		return "Success";
	case 1:		return "Invalid value";
	case 2:		return "Out of memory";
	case 3:		return "Not Initialized";
	case 4:		return "Deinitialized";

	case 100:	return "No device";
	case 101:	return "Invalid device";

	case 200:	return "Invalid image";
	case 201:	return "Invalid context";
	case 202:	return "Context already current";
	case 205:	return "Map failed";
	case 206:	return "Unmap failed";
	case 207:	return "Array is mapped";
	case 208:	return "Already mapped";
	case 209:	return "No binary for GPU";
	case 210:	return "Already acquired";
	case 211:	return "Not mapped";

	case 300:	return "Invalid source";
	case 301:	return "File not found";

	case 400:	return "Invalid handle";
	case 500:	return "Not found";
	case 600:	return "Not ready";

	case 700:	return "Launch failed";
	case 701:	return "Launch out of resources";
	case 702:	return "Launch timeout";
	case 703:	return "Launch incompatible texturing";

	case 999:	return "Unknown";
	};
	return "Unknown";
}


int main( int argc, char** argv) 
{
	if(argc<3)
	{
		puts("arguments: cubinname kernelname");
		return;
	}
	int length = 8;
	int cpu_output[8], cpu_output2[8];
	int size = sizeof(int)*length;
	CUdeviceptr gpu_output, gpu_output2;
	CUdevice device;
	CUcontext context;

	muRC(100, cuInit(0));
	muRC(95, cuDeviceGet(&device, 0));
	muRC(92, cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device));
	muRC(90, cuMemAlloc(&gpu_output, size));
	muRC(90, cuMemAlloc(&gpu_output2, size));

	CUmodule module;
	CUfunction kernel;
	CUresult result = cuModuleLoad(&module, argv[1]);
	muRC(0 , result);
	result = cuModuleGetFunction(&kernel, module, argv[2]);
	muRC(1, result); 
	//1 parameter as address of allocated memory of 128 bytes
	//1 thread launched
	//$(length) values expected
	int param = 0x1010;
	muRC(2, cuParamSetSize(kernel, 20));
	muRC(3, cuParamSetv(kernel, 0, &gpu_output, 8));
	muRC(3, cuParamSetv(kernel, 8, &gpu_output2, 8));
	muRC(3, cuParamSetv(kernel, 16, &param, 4));
	muRC(4, cuFuncSetBlockShape(kernel, 1,1,1));

	muRC(5, cuLaunch(kernel));

	muRC(6, cuMemcpyDtoH(cpu_output, gpu_output, size));
	muRC(6, cuMemcpyDtoH(cpu_output2, gpu_output2, size));
	muRC(7, cuCtxSynchronize());
	for(int i=0; i<length; i++)
	{
		printf("i=%i, output=%i\n", i, cpu_output[i]);
	}
	for(int i=0; i<length; i++)
	{
		printf("i=%i, output2=%i\n", i, cpu_output2[i]);
	}
	muRC(8, cuModuleUnload(module));
	muRC(9, cuMemFree(gpu_output));
	muRC(9, cuMemFree(gpu_output2));
	muRC(10, cuCtxDestroy(context));
	return 0;
}
