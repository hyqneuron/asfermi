Instructions follow the exact format of the output of cuobjdump. For more information please refer to [SourceFormat](SourceFormat.md).



## Direct Output Mode ##
Here's an example source file used for the "direct output mode" of asfermi.
```
!Kernel kernel1
!Param 4

MOV R1, c [0x1] [0x100];
MOV R0, c [0x0] [0x20];
MOV R2, 0xff;
ST  [R0], R2;
EXIT;

!EndKernel
```
This file is assembled with the following command line:
```
asfermi test.txt -o test.cubin
```


---

## Replace Mode ##
Here's an example of the source file used for the "replace mode" of asfermi.
```
MOV R2, 0x20;
LD R3, [R0];
```
It is assembled with the following command line:
```
asfermi test.txt -r target.cubin .text.targetkernelname 0x10 //replaces opcodes of targetkernelname at 0x10 and 0x18
```