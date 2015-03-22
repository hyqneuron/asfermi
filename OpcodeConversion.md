Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

## Conversion Instructions ##

#### F2I ####
Instruction usage:
```
F2I(.FTZ)(.dtype)(.stype)(.rnd) reg0, (-)composite operand;
```
Template opcode
```
0010 000100 1110 000000 010010 0000000000000000000000 0000000000 101000
        mod        reg0   mod2                  immea       mod3
```
|mod 1|meaning |
|:----|:-------|
|0    |default |
|1    |-composite operand|

|mod 2|.dtype |
|:----|:------|
|0    |.Uxx   |
|1    |.Sxx   |

|mod 3|meaning|
|:----|:------|
|0    |default|
|1    | |composite operand| |

|mod2 1:2|.stype |
|:-------|:------|
|10      |.F16   |
|01      |default(.F32)|
|11      |.F64   |

|mod2 4:5|.dtype |
|:-------|:------|
|10      |.x16   |
|01      |.x32   |
|11      |.x64   |

|mod3 2 |meaning|
|:------|:------|
|0      |default|
|1      |.FTZ   |

|mod3 7:8|.rnd  |
|:-------|:-----|
|10      |.FLOOR|
|01      |.CEIL |
|11      |.TRUNC|


---

#### F2F ####
Instruciton usage:
```
F2F(.FTZ).dtype.stype(.rop)(.SAT) reg0, (-/|)composite operand(|);
```
Template opcode
```
0010 000000 1110 000000 010010 0000000000000000000000 0000000000 001000
        mod        reg0   mod2                  immea       mod3
```
|mod 1|meaning|
|:----|:------|
|0    |default|
|1    |-composte operand|

mod 2 is effective only when both dtype and stype are F32
|mod 2|meaning|
|:----|:------|
|0    |.PASS  |
|1    |.ROUND |

|mod 3|meaning|
|:----|:------|
|0    |default|
|1    | |composte operand| |

|mod 4|meaning|
|:----|:------|
|0    |default|
|1    |.SAT   |

|mod 5|meaning|
|:----|:------|
|0    |default|
|1    |.S     |


|mod2 1:2|.stype |
|:-------|:------|
|10      |.F16   |
|01      |.F32   |
|11      |.F64   |

|mod2 4:5|.dtype |
|:-------|:------|
|00      |default|
|10      |.F16   |
|01      |.F32   |
|11      |.F64   |

|mod3 2 |meaning|
|:------|:------|
|0      |default|
|1      |.FTZ   |

|mod3 7:8|.rnd  |
|:-------|:-----|
|00      |default|
|10      |.RM|
|01      |.RP |
|11      |.RZ|


---

#### I2F ####
Instruction usage:
```
I2F(.dtype)(.stype)(.rnd) reg0, composite operand;
```
Template opcode
```
0010 000001 1110 000000 010010 0000000000000000000000 0000000000 011000
        mod        reg0   mod2                  immea       mod3
```

|mod 0|.stype|
|:----|:-----|
|0    |.Uxx  |
|1    |.Sxx  |

|mod 1|meaning |
|:----|:-------|
|0    |default |
|1    |-composite operand|

|mod 3|meaning|
|:----|:------|
|0    |default|
|1    | |composite operand| |

|mod2 1:2 |.stype|
|:--------|:-----|
|10       |.x16  |
|01       |.x32(default)|
|11       |.x64  |

|mod2 4:5 |.dtype|
|:--------|:-----|
|10      |.F16   |
|01      |default(.F32)|
|11      |.F64   |

|mod3 7:8|.rnd  |
|:-------|:-----|
|10      |.RM   |
|01      |.RP   |
|11      |.RZ   |


---

#### I2I ####
Instruction usage:
```
I2I.dtype.stype(.SAT) reg0, (-/|)composite operand(|);
```
Template opcode
```
0010 000000 1110 000000 010010 0000000000000000000000 0000000000 111000
        mod        reg0   mod2                  immea       mod3
```

|mod 0|.stype|
|:----|:-----|
|0    |.Uxx  |
|1    |.Sxx  |

|mod 1|meaning |
|:----|:-------|
|0    |default |
|1    |-composite operand|

|mod 2|.dtype|
|:----|:-----|
|0    |.Uxx  |
|1    |.Sxx  |

|mod 3|meaning|
|:----|:------|
|0    |default|
|1    | |composite operand| |

|mod 4|meaning|
|:----|:------|
|0    |default|
|1    |.SAT   |

|mod2 1:2 |.stype|
|:--------|:-----|
|00       |.X8   |
|10       |.X16  |
|01       |.X32  |
|11       |.X64  |

|mod2 4:5 |.dtype|
|:--------|:-----|
|00       |.X8   |
|10       |.X16  |
|01       |.X32  |
|11       |.X64  |

|mod3 1:2 |meaning|
|:--------|:------|
|00       |default|
|10       |.B1 on composite operand|
|01       |.B2 on composite operand|
|11       |.B3 on composite operand|

|mod3 9 |meaning|
|:------|:------|
|0      |default|
|1      |.CC on reg0|