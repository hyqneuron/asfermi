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
	if(argc>=4)
	{
		length = atoi(argv[3]);
	}
	int tcount = 1;
	if(argc>=5)
	{
		tcount = atoi(argv[4]);
	}
	int* cpu_output=new int[length];
	int size = sizeof(int)*length;
	int interval = 1;
	if(argc>=6)
	{
		interval = atoi(argv[5]);
	}
	bool odd = true;
	bool even = true;
	if(argc>=7)
	{
		int choice = atoi(argv[6]);
		if(choice==1)
			even = false;
		else if(choice==2)
			odd = false;
	}
	CUdeviceptr gpu_output;
	CUdevice device;
	CUcontext context;

	muRC(100, cuInit(0));
	muRC(95, cuDeviceGet(&device, 0));
	muRC(92, cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device));
	muRC(90, cuMemAlloc(&gpu_output, size));

	CUmodule module;
	CUfunction kernel;
	CUresult result = cuModuleLoad(&module, argv[1]);
	muRC(0 , result);
	result = cuModuleGetFunction(&kernel, module, argv[2]);
	muRC(1, result); 
	int param = 0x1010;
	muRC(2, cuParamSetSize(kernel, 20));
	muRC(3, cuParamSetv(kernel, 0, &gpu_output, 8));
	muRC(3, cuParamSetv(kernel, 16, &param, 4));
	muRC(4, cuFuncSetBlockShape(kernel, tcount,1,1));

	muRC(5, cuLaunch(kernel));

	muRC(6, cuMemcpyDtoH(cpu_output, gpu_output, size));
	muRC(7, cuCtxSynchronize());
	printf("length=%i\n", length);
	printf("tcount=%i\n", tcount);
	for(int i=0; i<length/interval; i++)
	{
		if(i%2==0)
		{
			if(!even) continue;
		}
		else
		{
			if(!odd) continue;
		}
		for(int j=0; j<interval; j++)
			printf("i=%i, j=%i, output=%i\n", i, j, cpu_output[i*interval+j]);
		if(interval!=1)
			puts("");
	}
	muRC(8, cuModuleUnload(module));
	muRC(9, cuMemFree(gpu_output));
	muRC(10, cuCtxDestroy(context));
	delete[] cpu_output;
	return 0;
}
