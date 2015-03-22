# Directives #

This page provides a list of directives that are supported/to be supported, as well as their descriptions. For usage syntax, see [SourceFormat](SourceFormat.md).

Directives are used to declare symbols and to send specific instructions to the assembler.




---

## Symbol Declarations ##
#### [0](Features#State_Numbers.md): Kernel ####
Description: the "Kernel" directive declares the start of a kernel, and it should always be used with another `"EndKernel"` directive. In between the region defined by the "Kernel" and `"EndKernel"` pair, the directive Param can be present to declare parameters.

Format:
```
!Kernel kernelname
!Param 4// one parameter sized at 4 bytes
!Param 4 2// another 2 parameters sized at 4 bytes each
... //instructions
!EndKernel
```
#### [0](Features#State_Numbers.md): Shared Memory ####
Description: the "Shared" directive declares the size of the shared memory that a kernel requires. The declaration must be made within a region defined by a "Kernel" and "EndKernel" pair.

Format
```
!Shared 0xabcd;
!Shared 128;
```
#### [0](Features#State_Numbers.md): Local Memory ####
Format
```
!Local 0xabcd;
!Local 128;
```
#### [0](Features#State_Numbers.md): Constant Memory ####
A set of directives are used to define constant memory: Constant2, Constant, EndConstant.

.nv.constant2 is the section in which ptxas places the user-defined constant symbols. While other constant sections may also be used, currently asfermi supports only .nv.constant2.

Constant2: defines the size of the .nv.constant2 section, that is, the total size of all constant memory.

Example:
```
!Constant2 0x40
!Constant int 0x0 //first argument indicates the type of the constant
                  //second argument indicates the offset of this object 
                  //from the beginning of the .nv.constant2 section
1, 2, 3, 4
!EndConstant

!Constant mixed 0x10
0x1001, F-1.3, FH1.909090909090, H10101010101010
!EndConstant
```
Supported types include int, long, float, double and mixed. The formatting of the constant expressions follow rules defined [here](SourceFormat#Immediate_Value.md)

The example above declares the size of the constant2 section as 0x40, then defines the values in 0x0 to 0x10 as 4 integers, and the values in 0x10 to 0x20 as a hex number, two floating point numbers and the higher 32 bits of an 64-bit integer.
#### [3](Features#State_Numbers.md): Texture Object ####


---

## Assembler Commands ##
#### [0](Features#State_Numbers.md): `RegCount` ####
Used within a kernel region to define the number of registers used by the kernel. However, if the actual number of registers required is larger than the defined number, the actual count will be used instead. Maximal count should not exceed 63.

Format:
```
!RegCount 12
!RegCount 0x10
```

#### [0](Features#State_Numbers.md): `BarCount` ####
Used within a kernel region to define the number of barriers used by the kernel. The defined count should not exceed 16. However, for the sake of experimentation a maximal count of 127 is actually allowed.

Format
```
!BarCount 12;
!BarCount 0x10;
```

#### [0](Features#State_Numbers.md): `RawInstruction` ####
This directive can be used to directly insert 4-byte words into a kernel, allowing the user to add instructions opcodes that are not yet supported by asfermi. 1 to 2 arguments are accepted. Each argument should be a 4-byte hexadecimal expression.

Format
```
!RawInstruction 0xabcd   //4 bytes
!RawInstruction 0xabcd 0xdef0 //8 bytes
```

#### [0](Features#State_Numbers.md): Machine ####
Specifies whether the cubin output will be 32-bit of 64-bit.

Format:
```
!Machine 32
//or
!Machine 64
```

#### [0](Features#State_Numbers.md): Arch ####
Specifies the architecture of the cubin output. Only effective in direct output mode. Any architecture specified through command lines will be overwritten.

Format:
```
!Arch sm_20
//or
!Arch sm_21
```

#### [0](Features#State_Numbers.md): Label ####
Description: the "Label" directive is used to mark the offset of a specific instruction so that the CAL instruction can refer to this label directly instead of to an offset value given by the programmer which may require some calculation and which may change when the source code is modified.

Format:
```
!Label LabelName
BRA !LabelName;
```
#### [0](Features#State_Numbers.md):Align ####
This directive aligns the following instruction to either 0x0 or 0x8 using NOPs.

Format:
```
!Align 0 //align to 0x**0
!Align 8 //align to 0x**8
```
#### [2](Features#State_Numbers.md): Unroll ####
Description: the "Unroll" directive can be used to the same instruction for many times. Note that the "Unroll" directive only operates on the first instruction that follows. Even if multiple instructions appear on the same line, only the first instruction will be repeated. `"BeginUnroll"` and `"EndUnroll"` can be used to repeat the instructions defined within this pair many times.

Format:
```
!Unroll 5 // 5 is the number of times of unrolling
...... //the single instruction that will be repeated

!Unroll 5 SearchTarget x y
... SearchTarget ... //SearchTarget will be replaced by a number x. x is incremented by y at every iteration.

!BeginUnroll 5 SearchTarget x y
...
...
!EndUnroll
```
#### [2](Features#State_Numbers.md): Set current parsers ####
Format:
```
!SetMasterParser Name
!SetInstructionParser Name
```
#### [3](Features#State_Numbers.md): Type-checking ####
#### [3](Features#State_Numbers.md): Macro ####
#### [3](Features#State_Numbers.md): Capitalize ####
Description: Defines whether the instruction parser should capitalize all letters of the input before it starts parsing. Can be enabled to support lower-case source code.

Format:
```
!Capitalie (on/off)
```