

### Interpretation ###
  * LEPC loads the address of the current instruction into reg0.
  * Fermi has limited memory access control. Because of this, one should be able to create dummy kernel images, modify them on-the-fly, then launch them from the host.


### Test ###
#### code ####
```
!Machine 64
!Kernel k
!Param 8 2
!Param 4

LEPC R0;
MOV R1, 0x1; //higher 32 bits of addr, got this by trial and error. 
LD.E.64 R0, [R0]; //load memory content at 64-bit address [R0|R1]
LEPC R2;
LEPC R3;
MOV R4, c[0X0][0X20]
MOV R5, c[0X0][0X24]
ST.E [R4+0X0], R0;
ST.E [R4+0X4], R1
ST.E [R4+0X8], R2
ST.E [R4+0Xc], R3

EXIT;
!EndKernel
```
#### result ####
```
length = 4
tcount=1
Device clock rate:1544000
time:0.026368

i=0, j=0, output=1c04
i=1, j=0, output=44000000
//0x00001c0444000000 is indeed opcode for LEPC R0;
i=2, j=0, output=a318
i=3, j=0, output=a320
```


### Test 2 ###
#### Code ####
the cubin:
```
!Machine 64
!Kernel k
!Param 8 2
!Param 4

MOV R15, 123; //the instruction to mod
LEPC R0; //R0-8=addr of previous instruction
MOV R1, 0x1; 
//store the opcode of "MOV R15, 99999" in R10 and R11
MOV32I R10, 0x7c03dde4
MOV32I R11, 0x2800c61a
IADD R0, R0, -8; //get the addr of the instruction to mod
ST.E.64 [R0], R10; //mod
LD.E.64 R6, [R0]; //load& check, not so meaningful due to cache
LEPC R2;
LEPC R3;
MOV R4, c[0X0][0X20] //load output memory pointer
MOV R5, c[0X0][0X24]
ST.E [R4+0X0], R6;
ST.E [R4+0X4], R7
ST.E [R4+0X8], R0
ST.E [R4+0Xc], R2
ST.E [R4+0X10], R3
ST.E [R4+0x14], R15 //output the result of R15

EXIT;
!EndKernel

```

The launcher (o.cu)
```
//please refer to http://code.google.com/p/asfermi/source/browse/test/main.cu for the utility functions before main()

int main( int argc, char** argv) 
{
	if(argc<3)
	{
		puts("arguments: cubinname kernelname length tcount interval choice");
		puts("	length: number of 4-byte elements to allocate in memory");
		puts("	tcount: number of threads");
		puts("	interval: number of output items per group");
		puts("	choice: 0, all; 1, odd group only; 2, even group only");
		return 0;
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
	bool repeat = true;
REP:
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
	if(repeat)
	{
		repeat = false;
		goto REP;
	}
	muRC(8, cuModuleUnload(module));
	muRC(9, cuMemFree(gpu_output));
	muRC(10, cuCtxDestroy(context));
	delete[] cpu_output;
	return 0;
}
```

#### result ####
```
Success at 100
Success at 95
Success at 92
Success at 90
Success at 0
Success at 1
Success at 2
Success at 3
Success at 3
Success at 4
Success at 5
Success at 6
Success at 7
length=6
tcount=1
i=0, j=0, output=2080628196
i=1, j=0, output=671139354
i=2, j=0, output=41728
i=3, j=0, output=41792
i=4, j=0, output=41800
i=5, j=0, output=123
Success at 1
Success at 2
Success at 3
Success at 3
Success at 4
Success at 5
Success at 6
Success at 7
length=6
tcount=1
i=0, j=0, output=2080628196
i=1, j=0, output=671139354
i=2, j=0, output=41728
i=3, j=0, output=41792
i=4, j=0, output=41800
i=5, j=0, output=99999 //result changed as intended
Success at 8
Success at 9
Success at 10

```