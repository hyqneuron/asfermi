Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

### Floating Point Instructions ###

#### FADD ####
Instruction Usage:
```
FADD(.FTZ)(.rnd)(.SAT) reg0, (-)(|)reg1(|), (-)(|)composite operand(|);
```
If the third operand is a floating point number, it will be a 32-bit floaitng point number with the last 12 bits truncated. If the negative sign in the immediate value is before 'F', then a modifier bit will be flipped. If the negative sign comes after the 'F', then the 20-bit immediate value will have the negative bit set.

Template opcode:
```
0000 000000 1110 000000 000000 0000000000000000000000 0000000000 001010
        mod        reg0   reg1                  immea       mod2
```
|mod bit| meaning  |
|:------|:---------|
|0      |0: default|
|       |1: -reg1  |
|1      |0: default|
|       |1: -composite operand|
|2      |0: default|
|       |1: |reg1| |
|3      |0: default|
|       |1: |composite operand| |
|4      |0: default|
|       |1: .FTZ   |

|mod2 0|meaning|
|:-----|:------|
|0     |default|
|1     |.FMA|

|mod2 1:2 | .rnd|
|:--------|:----|
|00|default (.rn) |
|10|.RM |
|01|.RP |
|11|.RZ |

|mod2 8|meaning|
|:-----|:------|
|0     |default|
|1     |.SAT   |


---

#### FADD32I ####
Instruction usage:
```
FADD32I(.FTZ) reg0, (-/|)reg1(|), 0xabcd;
```
Template opcode
```
0100 000000 1110 000000 000000 00000000000000000000000000000000 010100
        mod        reg0   reg1                           0xabcd
```
|mod bit| meaning  |
|:------|:---------|
|0      |0: default|
|       |1: -reg1  |
|2      |0: default|
|       |1: |reg1| |
|4      |0: default|
|       |1: .FTZ   |


---

#### FMUL ####
Instruction usage:
```
FMUL(.FTZ)(.rnd)(.SAT) reg0, reg1, (-)composite operand;
```
Template opcode:
```
0000 000000 1110 000000 000000 0000000000000000000000 0000000000 011010
        mod        reg0   reg1                  immea       mod2
```
|mod2 0|meaning|
|:-----|:------|
|0     |default|
|1     |-composite operand|

|mod2 1:2 | .rnd|
|:--------|:----|
|00|default(.rn)|
|10|.RM |
|01|.RP |
|11|.RZ |

|mod 0:1|.meaning|
|:------|:-------|
|00     |default|
|10     |.FMA  |
|10     |.FMA2 |
|11     |Invalid|


|mod 2:3|meaning|
|:------|:------|
|00     |default|
|10     |.FTZ   |
|01     |.FMZ   |
|11     |Invalid|

|mod 4|meaning|
|:----|:------|
|0    |default|
|1    |.SAT|



---

#### FMUL32I ####
Instruction usage:
```
FMUL32I(.FTZ)(.SAT) reg0, reg1, F(-)123.123;
```
Template opcode:
```
0100 000000 1110 000000 000000 00000000000000000000000000000000 001100
        mod        reg0   reg1                        fp number
```
|mod 0:1|meaning|
|:------|:------|
|00     |default|
|10     |.FMA   |
|10     |.FMA2  |
|11     |Invalid|


|mod 2:3|meaning|
|:------|:------|
|00     |default|
|10     |.FTZ   |
|01     |.FMZ   |
|11     |Invalid|

|mod 4|meaning|
|:----|:------|
|0    |default|
|1    |.SAT|


---

#### FFMA ####
Instruction Usage:
```
FFMA(.FTZ)(.roundOp)(.SAT) reg0, reg1, (-)composite operand, (-)reg3;
```

Or,

```
FFMA(.FTZ)(.roundOp)(.SAT) reg0, reg1, (-)reg2, (-)composite operand;
```

Note that either the third or the fourth operand is a composite operand, but the other one must be a register.

Template Opcode
```
0000 000000 1110 000000 000000 0000000000000000000000 0 000000  000 001100
        mod        reg0   reg1                  immea     reg3 mod2
```

|mod2 1:2 | .rnd|
|:--------|:----|
|00|default (.rn) |
|10|.RM |
|01|.RP |
|11|.RZ |

|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |-composite operand|

|mod 1|meaning|
|:----|:------|
|0    |default|
|1    |-reg3  |

|mod 2:3|meaning|
|:------|:------|
|00     |default|
|10     |.FTZ   |
|01     |.FMZ   |
|11     |Invalid|

|mod 4|meaning|
|:----|:------|
|0    |default|
|1    |.SAT|



---

#### FSETP ####
Instruction usage:
```
FSETP.compOp(.logicOp) p (,|p), (-)(|)reg1(|), (-)(|)composite operand(|) (,opp);
```
For what p, |p and opp mean, please refer to the setp instruction section in the ptx manual.

<a href='Hidden comment: 
issue: reg1 and composite operand should support operators - and ||
'></a>

Template opcode:
```
0000 000000 1110 111 000 000000 0000000000000000000000 0 1110 000000 00100
        mod       |p   p   reg1                  immea    opp   mod2
```

|mod bit|meaning   |
|:------|:---------|
|0      |0: default|
|       |1: -reg1  |
|1      |0: default|
|       |1: -composite operand|
|2      |0: default|
|       |1: |reg1| |
|3      |0: default|
|       |1: |composite operand| |

|mod2 0:3 value|.compOp|
|:-------------|:------|
|1             |.LT|
|2             |.EQ|
|3             |.LE|
|4             |.GT|
|5             |.NE|
|6             |.GE|
|7             |.NUM|
|8             |.NAN|
|9             |.LTU|
|10            |.EQU|
|11            |.LEU|
|12            |.GTU|
|14            |.GEU|

|mod2 4:5 value |.logicOp|
|:--------------|:-------|
|0              |.AND|
|1              |.OR|
|2              |.XOR|



---

#### FCMP ####
Instruction usage:
```
FCMP.CompOp(.FTZ) reg0, reg1, composite operand, reg3;
```

Template opcode
```
0000 000000 1110 000000 000000 0000000000000000000000 0 000000 0000 11100
        mod        reg0   reg1                  immea     reg3 mod2
```
|mod 4|meaning|
|:----|:------|
|0    |default|
|1    |.FTZ   |

|mod2 0:3 value|.CompOp|
|:-------------|:------|
|1             |.LT|
|2             |.EQ|
|3             |.LE|
|4             |.GT|
|5             |.NE|
|6             |.GE|
|7             |.NUM|
|8             |.NAN|
|9             |.LTU|
|10            |.EQU|
|11            |.LEU|
|12            |.GTU|
|14            |.GEU|



---

#### MUFU ####
Instruction usage:
```
MUFU.(Op)(.SAT) reg0, reg1;
```
Template opcode
```
0000 000000 1110 000000 000000 0000 0000000000000000000000000000 010011
        mod        reg0   reg1 mod2
```
|mod bit|meaning   |
|:------|:---------|
|0      |0: default|
|       |1: -reg1  |
|2      |0: default|
|       |1: |reg1| |
|4      |0: default|
|       |1: .SAT   |

|mod2 value|.Op    |
|:---------|:------|
|0         |.COS   |
|1         |.SIN   |
|2         |.EX2   |
|3         |.LG2   |
|4         |.RCP   |
|5         |.RSQ   |
|6         |.RCP64H|
|7         |.RSQ64H|



---

#### DADD ####
Instruciton usage
```
DADD(.rnd) reg0, (-)(|)reg1(|), (-)(|)composite operand(|);
```
Template opcode
```
1000 000000 1110 000000 000000 0000000000000000000000 0000000000 110010
        mod        reg0   reg1                  immea       mod2
```

|mod bit| meaning  |
|:------|:---------|
|0      |0: default|
|       |1: -reg1  |
|1      |0: default|
|       |1: -composite operand|
|2      |0: default|
|       |1: |reg1| |
|3      |0: default|
|       |1: |composite operand| |

|mod2 1:2 | .rnd|
|:--------|:----|
|00|default (.rn) |
|10|.RM |
|01|.RP |
|11|.RZ |


---

#### DMUL ####
```
DMUL(.rnd) reg0, reg1, (-)composite operand;
```
Template opcode
```
1000 000000 1110 000000 000000 0000000000000000000000 0000000000 001010
        mod        reg0   reg1                  immea       mod2
```

|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |-composite operand|

|mod2 1:2 | .rnd|
|:--------|:----|
|00|default (.rn) |
|10|.RM |
|01|.RP |
|11|.RZ |


---

#### DFMA ####
Instruction usage:
```
DFMA reg0, reg1, (-)composite operand, (-)reg3;
```

Or,

```
DFMA reg0, reg1, (-)reg2, (-)composite operand;
```


Note that either the third or the fourth operand is a composite operand, but the other one must be a register.

Template opcode
```
1000 000000 1110 000000 000000 0000000000000000000000 0 000000  000 000100
        mod        reg0   reg1                  immea     reg3 mod2
```

|mod 0|meaning|
|:----|:------|
|0    |default|
|1    |-composite operand|

|mod 1|meaning|
|:----|:------|
|0    |default|
|1    |-reg3  |


---

#### DSETP ####
Instruction usage:
```
DSETP.compOp(.logicOp) p (,|p), (-)(|)reg1(|), (-)(|)composite operand(|) (,opp);
```
```
1000 000000 1110 111 000 000000 0000000000000000000000 0 1110 000000 11000
        mod       |p   p   reg1                  immea    opp   mod2
```

|mod bit|meaning   |
|:------|:---------|
|0      |0: default|
|       |1: -reg1  |
|1      |0: default|
|       |1: -composite operand|
|2      |0: default|
|       |1: |reg1| |
|3      |0: default|
|       |1: |composite operand| |

|mod2 0:3 value|.compOp|
|:-------------|:------|
|1             |.LT|
|2             |.EQ|
|3             |.LE|
|4             |.GT|
|5             |.NE|
|6             |.GE|
|7             |.NUM|
|8             |.NAN|
|9             |.LTU|
|10            |.EQU|
|11            |.LEU|
|12            |.GTU|
|14            |.GEU|

|mod2 4:5 value |.logicOp|
|:--------------|:-------|
|0              |.AND|
|1              |.OR|
|2              |.XOR|