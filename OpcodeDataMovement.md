Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

### Data Movement Instructions ###


#### MOV ####
Instruction usage:
```
MOV(.S) reg0, composite operand;
```
If a hex number is used as the second operand, it cannot be longer than 20 bits.
Template opcode:
```
0010 011110 1110 000000 000000 0000000000000000000000 0000000000 010100
        mod        reg0   reg1                  immea
```

|mod 5|meaning|
|:----|:------|
|0    |default|
|1    |.S     |



---

#### MOV32I ####
Instruction usage:
```
MOV32I reg0, 0xabcd/1234;
```
Template opcode
```
0100 011110 1110 000000 000000 00000000000000000000000000000000 011000
                   reg0                                  0xabcd
```


---

#### LD ####
Instruction usage:
```
LD(.E)(.cop)(.type) reg0, [reg1(+0xabcd)];
```
Template opcode:
```
1010 000100 1110 000000 000000 00000000000000000000000000000000 0 00001
        mod        reg0   reg1                           0xabcd m
```
|mod 0:1 | .cop |
|:-------|:-----|
|00      |default(.ca)|
|10      |.CG   |
|01      |.CS   |
|11      |.CV   |

|mod 2:4 | .type |
|:-------|:------|
|000     | .U8     |
|100     | .S8     |
|010     | .U16    |
|110     | .S16    |
|001     |default(.u32)|
|101     |.64     |
|011     |.128    |

|m 0  |meaning|
|:----|:------|
|0    |default|
|1    |.E     |


---

#### LDU ####
Load uniform. Same as ldu in ptx.

Instruction usage:
```
LDU(.E)(.type) reg0, [reg1(+0xabcd)];
```
Template opcode
```
1010 000100 1110 000000 000000 00000000000000000000000000000000 0 10001
        mod        reg0   reg1                           0xabcd m
```
.type is the same as in LD

|m 0  |meaning|
|:----|:------|
|0    |default|
|1    |.E     |


---

#### LDL ####
Load local memory. Same as ld.local in ptx

Instruction usage:
```
LDL(.cop)(.type) reg0, [reg1(+0xabcd)];
```
Template opcode
```
1010 000100 1110 000000 000000 000000000000000000000000 00000000 000011
        mod        reg0   reg1                   0xabcd
```
.type is exactly the same as in LD. As for .cop, .CS is replaced with .LU instead.


---

#### LDS ####
Load shared memory. Same as ld.shared in ptx

Instruction usage:
```
LDS(.type) reg0, [reg1(+0xabcd)];
```

Template opcode
```
1010 000100 1110 000000 000000 000000000000000000000000 00000010 000011
        mod        reg0   reg1                   0xabcd
```
.type is the same as in LD.

Note that 0xabcd is a 24-bit signed integer. Its magnitude by right should not exceed 0xFFFF.


---

#### LDC ####
Load constant memory. Same as ld.const in ptx.

Instruction usage:
```
LDC(.type) reg0, c[0xa][0xbcde]
```
Template opcode
```
0110 000100 1110 000000 000000 0000000000000000 00000 00000000000 101000
        mod        reg0   reg1           0xbcde   0xa
```
.type is the same as in LD.

While it appears that 0xa could have a maximum of 0x1f, for now asfermi does not allow any value beyond 0xf. 0xbcde must less than 0x10000.


---

#### ST ####
Instruction usage:
```
ST(.E)(.cop)(.type) [reg1(+0xabcd)], reg0;
```
Template opcode:
```
1010 000100 1110 000000 000000 00000000000000000000000000000000 0 01001
        mod        reg0   reg1                           0xabcd m
```
For various values of mod 2:4 (.type) and their meaning please refer to the LD instruction above.

|mod 0:1 | .cop |
|:-------|:-----|
|00      |default(.wb)|
|10      |.CG   |
|01      |.CS   |
|11      |.WT   |

|m 0  |meaning|
|:----|:------|
|0    |default|
|1    |.E     |

---

#### STL ####
Store to local memory. Same as st.local in ptx.

Instruction usage:
```
STL(.cop)(.type) [reg1(+0xabcd)], reg0;
```
Template opcode
```
1010 000100 1110 000000 000000 000000000000000000000000 00000000 010011
        mod        reg0   reg1                   0xabcd
```
the meaning of mod(.cop and .type) is exactly same as in ST.


---

#### STS ####
Store shared memory. Same as st.shared in ptx

Instruction usage:
```
STS(.type) [reg1(+0xabcd)], reg0;
```

Template opcode
```
1010 000100 1110 000000 000000 000000000000000000000000 00000010 010011
        mod        reg0   reg1                   0xabcd
```
.type is the same as in LD.

Note that 0xabcd is a 24-bit signed integer. Its magnitude by right should not exceed 0xFFFF.



---

#### LDLK ####
Instruction usage:
```
LDLK(.type) p, reg0, [reg1 + 0xabcd];
```
Template opcode
```
1010 0001  00 1110 000000 000000 00000000000000000000000000000000   0 00101
      mod p_0        reg0   reg1                           0xabcd p_1
```
.type is the same as in LD.

p\_0 and p\_1 combined is p;

Note: While LDLK ought to operate on global memory, thus requiring the use of extended addressing mode in 64-bit environment, so far it seems LDLK does not support the .E modifier. As a result, LDLK may not work for 64-bit environments. (TBC)


---

#### LDSLK ####
Instruction usage:
```
LDSLK(.type) p, reg0, [reg1 + 0xabcd];
```
Template opcode
```
1010 000100 1110 000000 000000 000000000000000000000000 000 00000 100011
        mod        reg0   reg1                   0xabcd   p
```
.type is the same as in LD.


---

#### STUL ####
Instruction usage:
```
STUL(.type) [reg1+0xabcd], reg0;
```
Template opcode
```
1010 000100 1110 000000 000000 00000000000000000000000000000000 010111
        mod        reg0   reg1                           0xabcd
```
.type is the same as in LD.

Note: While STUL ought to operate on global memory, thus requiring the use of extended addressing mode in 64-bit environment, so far it seems STUL does not support the .E modifier. As a result, STUL may not work for 64-bit environments. (TBC)


---

#### STSUL ####
Instruction usage:
```
STSUL(.type) [reg1+0xabcd], reg0;
```
Template opcode
```
1010 000100 1110 000000 000000 000000000000000000000000 00000000 110011
        mod        reg0   reg1                   0xabcd
```
.type is the same as in LD.