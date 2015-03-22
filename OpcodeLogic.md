Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

### Logic And Shift Instructions ###


#### LOP ####
Instruction usage
```
LOP.op reg0, (~)reg1, (~)composite operand;
```
Template opcode
```
1100 000000 1110 000000 000000 0000000000000000000000 0000000000 010110
        mod        reg0   reg1                  immea       mod2
```
|mod 2:3 |meaning|
|:-------|:------|
|00      |.AND   |
|10      |.OR    |
|01      |.XOR   |
|11      |.PASS\_B|

|mod 1|meaning|
|:----|:------|
|0    |default|
|1    |~composite operand  |


|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |~reg1  |



---

#### LOP32I ####
Usage:
```
LOP32I.op reg0, (~)reg1, 0xabcd;
```
Template opcode
```
0100 000000 1110 000000 000000 00000000000000000000000000000000 011100
        mod        reg0   reg1                           0xabcd
```

|mod 2:3 |meaning|
|:-------|:------|
|00      |.AND   |
|10      |.OR    |
|01      |.XOR   |
|11      |.PASS\_B|

|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |~reg1  |


---

#### SHR ####
Instruction usage
```
SHR(.U32)(.W) reg0(.CC), reg1, composite operand;
```
Template opcode
```
1100 010000 1110 000000 000000 0000000000000000000000 0000000000 011010
        mod        reg0   reg1                  immea       mod2
```
|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |.W     |

|mod 4|meaning|
|:----|:------|
|0    |.U32   |
|1    |default|

|mod2 9|meaning|
|:-----|:------|
|0     |default|
|1     |.CC on reg0|


---

#### SHL ####
Instruction usage
```
SHL(.U32)(.W) reg0(.CC), reg1, composite operand;
```
Template opcode
```
1100 010000 1110 000000 000000 0000000000000000000000 0000000000 000110
        mod        reg0   reg1                  immea       mod2
```
mod and mo2 are the same as in SHR


---

#### BFE ####
Instruction usage:
```
BFE(.U32)(.BREV) reg0(.CC), reg1, composite operand
```
The composite operand should be a 16-bit number. The higher 8 bits indicate the length of the field, and the lower 8 bits indicate the starting bit of the field.

```
1100 010000 1110 000000 000000 0000000000000000000000 0000000000 001110
        mod        reg0   reg1                  immea       mod2
```

|mod 1|meaning|
|:----|:------|
|0    |default|
|1    |.BREV  |

|mod 4|meaning|
|:----|:------|
|0    |.U32   |
|1    |default|

|mod2 9|meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|


---

#### BFI ####
Instruction usage:
```
BFI reg0(.CC), reg1, composite operand, reg3;
```
Template opcode
```
1100 000000 1110 000000 000000 0000000000000000000000    0  000000  000 010100
        mod        reg0   reg1                 immea mod2   reg3 mod2 
```
|mod2 0|meaning|
|:-----|:------|
|0     |default|
|1     |reg0.CC|


---

#### SEL ####
Instruction usage:
```
SEL reg0, reg1, composite operand, (!)p;
```
Template opcode
```
0010 000000 1110 000000 000000 0000000000000000000000 0 0000 00000 000100
        mod        reg0   reg1                  immea      p
```