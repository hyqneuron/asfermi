Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

### Integer Instructions ###

#### IADD ####
Instruction usage:
```
IADD(.SAT)(.X) reg0, (-)reg1, (-)composite operand;
```
If the composite operand is an immediate value, it cannot be longer than 20 bits (cannot be negative).

Also note that the when the negative operator is used on both reg1 and reg3, a .PO modifier would appear in the cubin instead. What .PO does has not been tested.

Template opcode:
```
1100 000000 1110 000000 000000 0000000000000000000000 0000000000 010010
        mod        reg0   reg1                  immea
```

|mod 0:1|meaning|
|:------|:------|
|00     |default|
|10     |-reg3  |
|01     |-reg1  |
|11     |.PO    |

|mod bit|meaning   |
|:------|:---------|
|3      |0: default|
|       |1: .X     |
|4      |0: default|
|       |1: .SAT   |



---

#### IADD32I ####
Template opcode
```
0100 000000 1110 000000 000000 00000000000000000000000000000000 0 10000
        mod        reg0   reg1                           0xabcd m
```
|mod bit|meaning   |
|:------|:---------|
|3      |0: default|
|       |1: .X     |
|4      |0: default|
|       |1: .SAT   |

|m 0  |meaning|
|:----|:------|
|0    |default|
|1    |reg0.CC|


---

#### IMUL ####
Instruction Usage
```
IMUL(.U32/S32)(.U32/.S32)(.HI) reg0(.CC), reg1, composite operand;
```

Template opcode:
```
1100 010100 1110 000000 000000 0000000000000000000000 0000000000 001010
        mod        reg0   reg1                  immea       mod2
```
|mod bit| meaning|
|:------|:-------|
|2      |0: .U32, first modifier|
|       |1: .S32, first modifier|
|3      |0: .LO default         |
|       |1: .HI                 |
|4      |0: .U32, second modifier|
|       |1: .S32, second modifier|

|mod2 9|meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|



---

#### IMUL32I ####
Instruction usage:
```
IMUL32I(.U32/S32)(.U32/.S32)(.HI) reg0(.CC), reg1, composite operand;
```
Template opcode:
```
0100 010100 1110 000000 000000 00000000000000000000000000000000    0 01000
        mod        reg0   reg1                           0xabcd mod2
```

|mod bit| meaning|
|:------|:-------|
|2      |0: .U32, first modifier|
|       |1: .S32, first modifier|
|3      |0: .LO default         |
|       |1: .HI                 |
|4      |0: .U32, second modifier|
|       |1: .S32, second modifier|

|mod2 0|meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|


---

#### IMAD ####
Instruction usage:
```
IMAD(.U32/S32)(.U32/.S32)(.HI)(.SAT) reg0(.CC), (-)reg1, composite operand, (-)reg3;
```
Or,
```
IMAD(.U32/S32)(.U32/.S32)(.HI)(.SAT) reg0(.CC), (-)reg1, (-)reg2, composite operand;
```
Note that the when the negative operator is used on both reg1 and reg3, a .PO modifier would appear in the cubin instead. What .PO does has not been tested.

Either the third or the fourth operand is a composite operand, but the other one must be a register.

Template opcode:
```
1100 010100 1110 000000 000000 0000000000000000000000 0 000000  000 000100
        mod        reg0   reg1                  immea m   reg3 mod2
```

|mod 0:1|meaning|
|:------|:------|
|00     |default|
|10     |-reg3  |
|01     |-reg1  |
|11     |.PO    |

|2      |0: .U32, first modifier|
|:------|:----------------------|
|       |1: .S32, first modifier|
|3      |0: .LO default         |
|       |1: .HI                 |
|4      |0: .U32, second modifier|
|       |1: .S32, second modifier|

|m 0   |meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|

|mod2 1| meaning |
|:-----|:--------|
|0     |default|
|1     |.SAT   |



---

#### ISCADD ####
Instruction usage:
```
ISCADD reg0, reg1, composite operand, int1;
```
reg0 = reg1<<int1 + composite operand;

Note that int1 must be less than 32.

Template opcode:
```
1100 0 00000 1110 000000 000000 0000000000000000000000 0000000000 000010
        int1        reg0   reg1                  immea       mod2
```

|mod2 1:2|meaning               |
|:-------|:---------------------|
|00      |0: default            |
|01      |1: -reg1              |
|10      |0: -composite operand |
|11      |1: .PO                |

|mod2 9|meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|



---

#### ISETP ####
Instruction usage:
```
ISETP.CompOp(.U32)(.LogicOp) p, |p, reg0, composite operand, opp;
```
Template opcode:
```
1100 010000 1110 111 000 000000 0000000000000000000000 0 1110 000000 11000
        mod       |p   p   reg0                  immea    opp   mod2
```

|mod 4|meaning|
|:----|:------|
|0    |.U32   |
|1    |default|

|mod2 0:3 value|`.CompOp`|
|:-------------|:--------|
|1             |.LT|
|2             |.EQ|
|3             |.LE|
|4             |.GT|
|5             |.NE|
|6             |.GE|

|mod2 4:5 value |`.LogicOp`|
|:--------------|:---------|
|0              |.AND|
|1              |.OR|
|2              |.XOR|


---

#### ICMP ####
Instruction usage:
```
ICMP.CompOp(.U32) reg0, reg1, composite operand, reg3
```
reg0 = (reg3 `CompOp` 0)? reg1 : composite operand; The .U32 modifier indicates the type of reg3. reg3 is assumed to be S32 when .U32 is not specified.

Template opcode
```
1100 010000 1110 000000 000000 0000000000000000000000 0 000000  000 001100
        mod        reg0   reg1                  immea     reg3 mod2
```
|mod 4|meaning|
|:----|:------|
|0    |.U32   |
|1    |default(.s32)|

|mod2 0:2 value|`.CompOp`|
|:-------------|:--------|
|1             |.LT|
|2             |.EQ|
|3             |.LE|
|4             |.GT|
|5             |.NE|
|6             |.GE|


---

#### VADD ####
Instruction Usage:
```
VADD(.UD)(.Op1Type)(.Op2Type)(.SAT)(.op)(.S) reg0(.CC), (-)reg1(.r1sel), (-)composite operand(.r2sel), reg3;
```
Note that the composite operand of VADD should be either a register or a 16-bit integer. When a 16-bit integer is used, .r2sel should not be present.

While in PTX dtype (please refer to ptx manual) must be specified as u32 or s32, for VADD the dtype of s32 is assumed. if u32 is the desired type, use the .UD modifier.

Also, while in PTX atype and btype can only be u32/s32, for VADD Op1Type and Op2Type can be Uxx/Sxx, where xx=8,16 or 32. When xx!=32, sub-word selection of the corresponding operand takes place in accordance with the r1sel and r2sel modifiers.

Also, even though the immediate bit field is 16-bit in length, cuobjdump always outputs Op2Type as U8/S8 when the immediate operand is used, even when the immediate operand has more than 8 effective bits.

Template opcode:
```
0010 011000 1110 000000 000000 0000000000000000 1000000 000000  111 000011
        mod        reg0   reg1        composite    mod3   reg3 mod4
```

|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |.SAT   |

|mod 1|meaning|
|:----|:------|
|0    |default|
|1    |-reg1  |

|mod 2|meaning|
|:----|:------|
|0    |default|
|1    |-composite operand|

Note that when both reg1 and composite operand are set to negative, .PO appears instead in the disassembly of cuobjdump.

|mod 3|.Op1Type|
|:----|:-------|
|0    |.Uxx, |
|1    |.Sxx  |

|mod 4|.Op2Type|
|:----|:-------|
|0    |.Uxx, |
|1    |.Sxx  |

With asfermi, default OpTypes are S32. When only 1 OpType modifier is given it will be treated as Op1Type and Op2Type will be assumed to be S32, unless Op2 is actually an immediate operand, in that case Op2Type will become S8 if not otherwise specified.

|mod 5|meaning|
|:----|:------|
|0    |default|
|1    |.S     |

|mod3 6|meaning|
|:-----|:------|
|0     |.UD    |
|1     |default|

|mod3 2:4 value |.Op1Type, .r1sel |
|:--------------|:----------------|
|0              |.x8, default    |
|1              |.x8, .B1        |
|2              |.x8, .B2        |
|3              |.x8, .B3        |
|4              |.x16, default   |
|5              |.x16, .H1       |
|6              |.x32, default   |
|7              |invalid         |

x=S/U

|mod3 1|meaning|
|:-----|:------|
|0     |use composite as 0xabcd|
|1     |use composite as reg2, specified below|

mod2 = composite 7:9

reg2 = composite 10:15

|mod2 value |.Op2Type, .r2sel|
|:----------|:---------------|
|0          |.x8, default    |
|1          |.x8, .B1        |
|2          |.x8, .B2        |
|3          |.x8, .B3        |
|4          |.x16, default   |
|5          |.x16, .H1       |
|6          |.x32, default   |
|7          |invalid         |


|mod3 0|meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|

|mod4 0:2 value |.op     |
|:--------------|:-------|
|0              |.MRG\_16H|
|1              |.MRG\_16L|
|2              |.MRG\_8B0|
|3              |.MRG\_8B2|
|4              |.ACC    |
|5              |.MIN    |
|6              |.MAX    |
|7              |default |

VADD does not have MRG\_8B1 or MRG\_8B3. If PTX specifies dsel=b1/b3, an extra PRMT instruction is used.