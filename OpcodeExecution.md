Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

### Execution Control Instructions ###


#### SCHI ####
SCHI is a sm\_30-only pseudo-instruction located at the beginning of each 0x40 byte-sized instruction block of sm\_30's kernel. It controls the dispatch interval of the following 7 instructions.

Instruction usage
```
SCHI op0, op1, op2, op3, op4, op5, op6;
INST0
INST1
INST2
INST3
INST4
INST5
INST6
```
where op(x) is either a hex number or a binary number (least-significant bit first) not exceeding 0xff or 11111111, and INST(x) may be any instruction. op(x) controls the dispatch interval between INST(x) and INST(x+1)

Template opcode
```
1110 00000000 00000000 00000000 00000000 00000000 00000000 00000000 0100
     op0      op1      op2      op3      op4      op5      op6
```


---

#### SSY ####
Instruction usage
```
SSY const mem addr/(-)0xabcd/!labelName;
```
Template opcode
```
1110 000000 0000 000000 000000 000000000000000000000000 00000000 000110
                   mod2           const mem addr/0xabcd
```

|mod2 5|meaning|
|:-----|:------|
|0     |0xabcd |
|1     |const mem addr|


---

#### BRA ####
Instruction usage
```
BRA(.U)(.LMT)  const mem addr/(-)0xabcd/!labelName;
```
For the usage of labels, refer to [Directives#:\_Label](Directives#:_Label.md).

Template opcode
```
1110 011110 1110 000000 000000 000000000000000000000000 00000000 000010
                   mod2           const mem addr/0xabcd
```
|mod2 3|meaning|
|:-----|:------|
|0     |default|
|1     |.LMT   |

|mod2 4|meaning|
|:-----|:------|
|0     |default|
|1     |.U     |

|mod2 5|meaning|
|:-----|:------|
|0     |0xabcd |
|1     |const mem addr|



---

#### CAL ####
Instruction usage:
```
CAL(.NOINC) const mem addr/(-)0xabcd/!labelName;
```
0xabcd is the address relative to the address of the instruction right after the CAL instruction. Note that the cuobjdump would output the absolute address of the target instruction in the kernel instead of the relative address. 0xabcd cannot 24 bits in length.

Template opcode:
```
1110 000000 0000 001000 000000 000000000000000000000000 00000000 001010
                   mod2                          0xabcd
```
|mod2 bit|value and meaning|
|:-------|:----------------|
|3       |1: default|
|        |.NOINC    |
|5       |0: default|
|        |1: constant memory|


---

#### PRET ####
Set return address for the RET instruction

Instruction usage:
```
PRET  const mem addr/(-)0xabcd/!labelName;
```

Template code
```
1110 000000 0000   01 0000 000000 000000000000000000000000 00000000 011110
                 mod2                const mem addr/0xabcd
```
|mod2 0|meaning|
|:-----|:------|
|0     |.NOINC |
|1     |default|

|mod2 1|meaning|
|:-----|:------|
|0     |0xabcd|
|1     |const mem addr|

---

#### RET ####
Template opcode:
```
1110 011110 1110 000000 000000 00000000000000000000000000000000 001001
```


---

#### JMP ####
Instruction usage:
```
JMP(.U)(.LMT) const mem addr/(-)0xabcd/!labelName;
```
Template opcode
```
1110 011110 1110 000000 000000 000000000000000000000000 00000000 000000
                 mod2            const mem addr/0xabcd
```
|mod2 3|meaning|
|:-----|:------|
|0     |default|
|1     |.LMT   |

|mod2 4|meaning|
|:-----|:------|
|0     |default|
|1     |.U     |

|mod2 5|meaning|
|:-----|:------|
|0     |0xabcd |
|1     |const mem addr|


---

#### JCAL ####
Instruction usage:
```
JCAL(.NOINC) const mem addr/(-)0xabcd/!labelName;
```
Template opcode
```
1110 000000 0000 001000 000000 000000000000000000000000 00000000 001000
                 mod2             const mem addr/0xabcd
```
|3       |1: default|
|:-------|:---------|
|        |.NOINC    |
|5       |0: default|
|        |1: constant memory|


---

#### EXIT ####
Template opcode:
```
1110 011110 1110 000000 000000 00000000000000000000000000000000 000001
```


---

#### NOP ####
Instruction usage:
```
NOP(.TRIG)(.Op)(.S) (CC(.CCop)) (, 0xabcd);
```
Template opcode:
```
0010 011110 1110 000000 000000 0000000000000000 00000000 00000000 000010
        mod                    0xabcd                        mod2
```
Note: the following modifiers are produced according to the output of cuobjdump. Whether they are meaningful to NOP or not is not confirmed. Conditional code operation seems to be supported by NOP, BRA, RET, EXIT, BRK, CONT, JMP, LONGJMP, but such an operand is not supported by the listed instructions apart from NOP.

|mod 0:4 value|.CCop|
|:------------|:----|
|0            |.F   |
|1            |.LT  |
|2            |.EQ  |
|3            |.LE  |
|4            |.GT  |
|5            |.NE  |
|6            |.GE  |
|7            |.NUM  |
|8            |.NAN  |
|9            |.LTU  |
|10           |.EQU  |
|11           |.LEU  |
|12           |.GTU  |
|13           |.NEU  |
|14           |.GEU  |
|15           |.T  |
|16           |.OFF  |
|17           |.LO  |
|18           |.SFF  |
|19           |.LS  |
|20           |.HI  |
|21           |.SFT  |
|22           |.HS  |
|23           |.OFT  |
|24           |.CSM\_TA  |
|25           |.CSM\_TR  |
|26           |.CSM\_MX  |
|27           |.FCSM\_TA  |
|28           |.FCSM\_TR  |
|29           |.FCSM\_MX  |
|30           |.RLE  |
|31           |.RGT  |

|mod 5|meaning|
|:----|:------|
|0    |default|
|1    |.S     |

|mod2 7|meaning|
|:-----|:------|
|0     |default|
|1     |.TRIG  |

|mod2 3:6 value|.Op |
|:-------------|:---|
|0             |none  |
|1             |.FMA64|
|2             |.FMA32|
|3             |.XLU  |
|4             |.ALU  |
|5             |.AGU  |
|6             |.SU   |
|7             |.FU   |
|8             |.FMUL |


---

#### PBK ####
Instruction usage:
```
PBK constant mem address/0xabcd;
```
Template opcode
```
1110 000000 1110 0 00000 000000 000000000000000000000000 00000000 010110
                 m                     const addr/0xabcd
```
|m|meaning|
|:|:------|
|0 |0xabcd |
|1 |const addr|


---

#### BRK ####
Instruction usage:
```
BRK;
```
Template opcode
```
1110 01110 1110 000000 000000 00000000000000000000000000000000 010101
```



---

#### PCNT ####
Instruction usage:
```
PCNT const mem addr/0xabcd;
```
Template opcode:
```
1110 000000 1110 0 00000 000000 000000000000000000000000 00000000 001110
                 m                     const addr/0xabcd
```

|m|meaning|
|:|:------|
|0 |0xabcd |
|1 |const addr|


---

#### CONT ####
Instruction usage:
```
CONT;
```
Template opcode
```
1110 011110 1110 000000 000000 00000000000000000000000000000000 001101
```


---

#### PLONGJMP ####
Instruction usage:
```
PLONGJMP const mem addr/0xabcd;
```
Template opcode
```
1110 000000 1110 000000 000000 000000000000000000000000 00000000 011010
                   mod2               const addr/0xabcd
```

|mod2 5|meaning|
|:-----|:------|
|0     |0xabcd |
|1     |const addr|


---

#### LONGJMP ####
Instruction usage:
```
LONGJMP;
```
Template opcode
```
1110 011110 1110 000000 000000 00000000000000000000000000000000 010001

```


---

### Synchronization, Atomic Instructions ###
#### BAR ####
Instruction usage:
```
bar.sync:         BAR.RED.POPC RZ,    bar (,tcount);
bar.red.popc.u32: BAR.RED.POPC reg0,  bar (,tcount), (!)c;
bar.red.op.pred:  BAR.RED.Op   RZ, p, bar (,tcount), (!)c;
bar.arrive:       BAR.ARV             bar, tcount;
```

  * reg0: the register to which the number of true predicates is to be stored at the end of the reduction operation.
  * bar: A numerical(0xa)/register expression that identifies the barrier (0 to 15).
  * tcount: A numerical(0xabc)/register expression indicating the number of threads to reach the instruction before reinitialization of barrier. When unspecified, all threads of the block will have to reach the barrier before the barrier is reinitialized.
  * p,c: please refer to ptx manual for more information.

Template opcode
```
0010 000000 1110 000000 000000 111111000000 00000000   00 0 1110 111 00 001010 
        mod        reg0    bar       tcount          mod2      c   p
```

|mod 2|meaning|
|:----|:------|
|0    |.RED   |
|1    |.ARV   |

|mod 3:4|meaning|
|:------|:------|
|00     |.POPC  |
|10     |Op: AND|
|01     |Op: OR |
|11     |invalid|

|mod2 0|meaning for bar|
|:-----|:--------------|
|0     |use reg1|
|1     |use reg1(0xa)|

|mod2 1|meaning for tcount|
|:-----|:-----------------|
|0     |use reg2|
|1     |use reg2(0xabc)|



---

#### B2R ####
Instruction usage:
```
B2R(.Op)(.S) reg0, 0xab;
```
  * 0xab is limited to 6-bit.

Template opcode
```
0010 000000 1110 000000 000000 00000000000000000000000000000000 011100
        mod        reg0   0xab
```
|mod 0:1|.Op|
|:------|:--|
|00     |default|
|10     |.XLU   |
|01     |.ALU   |
|11     |invalid|

|mod 5|meaning|
|:----|:------|
|0    |default|
|1    |.S     |

---

#### MEMBAR ####
Instruction usage:
```
MEMBAR.Lvl;
```
Template opcode
```
1010 000000 1110 000000 000000 00000000000000000000000000000000 000111
        mod
```

|mod 3:4|.Lvl|
|:------|:---|
|00     |.CTA|
|10     |.GL |
|01     |.SYS|
|11     |invalid|


---

#### ATOM ####
Instruction usage:
```
ATOM(.E).Op(.Type) reg3, [reg1+0xabcd], reg0 (,reg4);
```
0xabcd can be at most 20-bit long.

Template opcode
```
1010 000000 1110 000000 000000 00000000000000000 000000 000000      000 0010 10
        mod        reg0   reg1          0xabcd_0   reg3   reg4 0xabcd_1 mod2
```
0xabcd\_0 and abcd\_1 together gives 0xabcd.

|mod 1:4 value|.Type|
|:------------|:----|
|0            |.ADD |
|1            |.MIN |
|2            |.MAX |
|3            |.INC |
|4            |.DEC |
|5            |.AND |
|6            |.OR  |
|7            |.XOR |
|8            |.EXCH|
|9            |.CAS |
|10 and above |invalid|

|mod2 3|meaning|
|:-----|:------|
|0     |default|
|1     |.E     |

mod 0 together with mod2 0:2 gives mod3

|mod3 value|meaning|
|:---------|:------|
|4         |default|
|5         |.U64   |
|7         |.S32   |
|11        |.F32.FTZ.RN|
|others    |invalid|

Note that:
  * For CAS and EXCH, b64 is just U64.
  * .FTZ.RN (flush subnormal numbers to sign-preserved zero, round to nearest even) is the default operation mode for ATOM.ADD.F32. While cuobjdump always outputs .FTZ.RN along with .F32 for ATOM, when working with asfermi, .FTZ.RN can omitted.


---

#### RED ####
Instruction usage:
```
RED(.E).Op(.Type) [reg1+0xabcd], reg0;
```
Template opcode
```
1010 000000 1110 000000 000000 00000000000000000000000000000000 0010 00
        mod        reg0   reg1                           0xabcd mod2
```
All the modifier bits are the same as in ATOM. The only difference is that .EXCH and .CAS are not applicable to RED.


---

#### VOTE ####
Instruction usage:
```
vote.all:    VOTE.ALL RZ,   p0, (!)p1;
vote.any:    VOTE.ANY RZ,   p0, (!)p1;
vote.uni:    VOTE.EQ  RZ,   p0, (!)p1;
vote.ballot: VOTE.ANY reg0, pt, (!)p1;
Unconfirmed modes:
VOTE.VTG.R  0xabcd;
VOTE.VTG.A  0xabcd;
VOTE.VTG.RA 0xabcd;
```
Template opcode
```
0010 000000 1110 000000 0000 00 0000000000000000000000000000 000 0 010010
        mod        reg0   p1                          0xabcd  p0
```
|mod 2:4 value|.Mode  |
|:------------|:------|
|0            |.ALL   |
|1            |.ANY   |
|2            |.EQ(.uni)|
|5            |.VTG.R |
|6            |.VTG.A |
|7            |.VTG.RA|