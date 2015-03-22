

#### How to build asfermi ####
The makefile is /trunk/generated.txt
If you need windows binary please contact the author at hyq.neuron _at_ gmail.com

#### How to Execute Cubin Kernels ####
The standard approach is to use CUDA driver API calls to load the cubin module and then launch the desired kernel. The following code snippet demonstrates how to do that:
```

void muRC(int position, CUresult result)
{
	if(result==0)
		printf("Success at %i\n", position);
	else
		printf("Error at %i:%s\n", position, muGetErrorString(result)); 
//definition of muGetErrorString can be found at http://code.google.com/p/asfermi/source/browse/test/main.cu
}
int main()
{
	CUdeviceptr gpu_output;
	CUdevice device;
	CUcontext context;
	int size = SIZE_OF_ALLOCATION;

	muRC(100, cuInit(0));
	muRC(95, cuDeviceGet(&device, 0));
	muRC(92, cuCtxCreate(&context, CU_CTX_SCHED_SPIN, device));
	muRC(90, cuMemAlloc(&gpu_output, size));

	CUmodule module;
	CUfunction kernel;
	CUresult result = cuModuleLoad(&module, "cubin_file_name");
	muRC(0 , result);
	result = cuModuleGetFunction(&kernel, module, "kernel_name");
	muRC(1, result); 
	int param = 0x1010;
	//specify the parameters to be passed to the kernel
	muRC(2, cuParamSetSize(kernel, 20));
	muRC(3, cuParamSetv(kernel, 0, &gpu_output, 8));
	muRC(3, cuParamSetv(kernel, 16, &param, 4));
	//launch
	muRC(4, cuFuncSetBlockShape(kernel, tcount,1,1));

	muRC(5, cuLaunch(kernel));

	muRC(6, cuMemcpyDtoH(cpu_output, gpu_output, size));
	muRC(7, cuCtxSynchronize());
	// process output
	// ...
	muRC(8, cuModuleUnload(module));
	muRC(9, cuMemFree(gpu_output));
	muRC(10, cuCtxDestroy(context));
	delete[] cpu_output;
	return 0;
}
```
For more information please refer to CUDA API reference.

#### Special registers ####
```
%tid.x:   S2R Rx, SR_Tid_X;
%tid.y:   S2R Rx, SR_Tid_Y;
%tid.z:   S2R Rx, SR_Tid_Z;
%landid:  S2R Rx, SR_LandId;
%ctaid.x: S2R Rx, SR_CTAid_X;
%ctaid.y: S2R Rx, SR_CTAid_Y;
%ctaid.z: S2R Rx, SR_CTAid_Z;
%smid:    S2R Rx, SR_VirtId;
          BFE Rx, Rx, 0x914;
%nsmid:   S2R Rx, SR_VirtCfg;
          BFE Rx, Rx, 0x914;
%clock:   S2R Rx, SR_ClockLo;
```
Note that clock number obtained is scheduler clock number.

More special registers: [OpcodeMiscellaneous#S2R](OpcodeMiscellaneous#S2R.md)



#### `MOV R1, c[0x1][0x100]` ####
This instruction appears at the beginning of all ptxas/nvcc-generated cubins. `C[0x1][0x100]` is the initial stack top and local window top. NVCC/PTXAS use its content to initialize the virtual SP register (and of course to access local variables) in global functions and [R1](https://code.google.com/p/asfermi/source/detail?r=1) is always the current virtual SP passed to device function.  the Local window/stack is alway 32-bit (actually it's 24-bit) in both 32-bit/64-bit device code. the stack is top-down growing.
(description contributed by Sun HuanHuan)

#### `MOV R0, c[0x0][0x20]` ####
A kernel's parameter window starts at `c[0x0][0x20]` and can be at most 256-byte wide. The instruction `MOV R0, c[0x0][0x20]` Loads the first 4 bytes of the parameters passed in to `R0`.

#### RZ ####
RZ means either 'no register' or 'value zero'.

#### pt ####
pt means either 'no predicate' or 'predicate true'

#### Working with 64-bit platform ####
A full address is 64-bit and requires 2 MOV to complete the loading. ST.E will then be used to store any value to an address. Eg:
```
//store 0x1234 to address specified in the first 8-byte parameter
MOV R0, c[0x0][0x20];
MOV R1, c[0x0][0x24];
MOV R2, 0x1234;
ST.E [R0], R2;
```
Note that ST.E [R1](R1.md) will almost always generate kernel launch error. The reason for this is not yet known.

#### How to get clock count ####
See [#Special\_registers](#Special_registers.md)

#### Instruction not supported ####
When trying to reassemble the output of cuobjdump, if asfermi says that some instruction is not supported, here's what you can do:
  1. Use other instructions to simulate it
  1. Copy the hex dump of the instruction and use the [RawInstruction](Directives#:_RawInstruction.md) directive
  1. Email me and tell me that you want to see the support for that instruction.