# Source Code Format #

This file specifies the format that should be taken by the source assembly code. For some sample code, please see [CodeExample](CodeExample.md).



General rules:
  * The source file sent to the assembler should only contain 3 types of lines:
    * Directive
    * Instruction
    * Comment
  * Everything is case-sensitive
  * If a line contains a directive, it must not contain any other directives or instructions. However, comments could be inserted right after the directive.


---

## Directives ##
Use:
  * Declare symbols: kernel (parameters), texture symbol, constant memory, shared memory, external symbols
  * Send specific commands to the assembler
For a list of directives that are supported/to be supported, see [Directives](Directives.md).

Format:
```
!directive_name argument1 argument2...
```
Note: labels are written as directives.

[State](Features.md): 0

---

## Instructions ##
Instructions take the same format as the output of cuobjdump. For a list of instructions that are supported/to be supported, see [Instructions](Instructions.md).

Format:
```
//Instead of square brackets, round brackets are used to represent
//'optional' so as to avoid confusion with memory operands.
(@Px) NAME OP1 (,OP2 (,OP3))

NAME.MOD1.MOD2 OP1(.MOD3) (, OP2(,OP3))
NAME; NAME OP1, OP2; NAME
    @P0 NAME OP1, OP2
```

General rules:
  * Everything should be upper-case
  * The assembler does not enforce the use of the semi-colon. It could be used in a line to separate one instruction from another
  * @Px is predicate. If it exists, space has to be used to separete it from the instruction name.
  * Instruction name must be separated from operands using blank spaces or tabs.
  * Operands are separated from each other using commas.
  * A modifier is prefixed with a dot.


asfermi does syntax checking in a very slack manner. The following code demonstrates this:
```
LD R0, [R2  + 0x10]  //fine
LD R0, [R2  + 0x10]; //fine
LD R0xxXX, [R2  + 0x10] //fine, the xxXX is just ignored
LD R0 is the value, [R2 is another value + 0x10kkk //fine
```
This feature may be used to conveniently annotate the source. For example:
```
LD R0_xvalue [R2_xaddr];
```
I may write a GUI editor to facilitate coding using this feature when asfermi is more complete.



### Operands ###
#### Registers ####
Basically, registers can range between [R0](https://code.google.com/p/asfermi/source/detail?r=0) to [R62](https://code.google.com/p/asfermi/source/detail?r=62). RZ represents a constant value of 0. It is okay to put non-numeric characters after the register name. If the register is greater than or equal to [R10](https://code.google.com/p/asfermi/source/detail?r=10), even numeric characters can directly follow the register name (eg. [R3000](https://code.google.com/p/asfermi/source/detail?r=3000) is taken as [R30](https://code.google.com/p/asfermi/source/detail?r=30)).
#### Immediate Value ####
For now, asfermi accepts hexadecimal, floating point or integer number expressions as immediate values.
Hexadecimal value:
```
0x12345678
```
Hex value must start with '0x' or '0X'. Not all eight digits need to be present. For example, 0x001 would be legal. Also, characters after the eighth digit are ignored. For example, 0x123456789 is taken as 0x12345678. Additionally, characters starting from the first non-hexadecimal character will be ignored. For example, 0x10annn is taken as 0x10a.

Floating point numbers:
```
32-bit: F123.456 or F-123.345;
Higher 32 bits of 64-bit number: FH123.456 or FH-333.2;
Lower 32 bits of 64-bit number: FL123.456
```
Integers:
```
32-bit: 12345 or -234;
Higher 32 bits of 64-bit number: H12345
Lower 32 bits of 64-bit number: L12345
```
Note:

asfermi uses C++'s library functions, [atoi](http://www.cplusplus.com/reference/clibrary/cstdlib/atoi/), [atol](http://www.cplusplus.com/reference/clibrary/cstdlib/atol/) and [atof](http://www.cplusplus.com/reference/clibrary/cstdlib/atof/), to convert numerical constant expressions. For for information about the conversion, please visit the three links given.

Not all instructions that accept immediate operands support all the three types above. For example, FADD32I accepts only hex values and floating point numbers, while IADD32I accepts hex values and integer numbers. Additionally, some instructions, such as FADD and IADD, accept only 20-bit immediate values. 20-bit floating point immediate is a 32-bit floaitng point number with the last 12 bits truncated.

Also, while integer and floating point numbers can have negative signs, hexadecimal values cannot. Furthermore, a number and its negative sign should not be separated by spaces.
#### Global Memory ####
A global memory operand has the following format:
```
[reg1 + 32-bit hex number]
```
Both register and the hex address can be present, and at least one of them should be present. When only the register is present, the immediate value will take the default of 0x0. When only the immediate value is present, the register will take the default value of RZ.

The register and immediate value format follow the exact format described in sections above. While the opening square bracket is needed, the closing bracket is not.

State: 0
#### Constant Memory ####
Format:
```
c[0xaa][reg1+0xbb]
```
0xaa is the constant memory bank number. It can range from 0x0 to 0xa. reg1+0xbb is the constant memory address inside the specified bank. Note that 0xbb cannot be larger than 0xFFFF. For instructions that do not specifically operate on constant memory, such as MOV and FADD, the register should not be present.

#### Composite operand ####
Some instructions support an operand that can take one of the three types:
  1. register
  1. 20-bit immediate value
  1. constant memory without the register component


---

## Comments ##
Comments are exactly the same as as in C/C++.

Format:
```
//in-line comment

/*
block comment
*/
```
State: 0