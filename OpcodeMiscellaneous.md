Note: round brackets, instead of square brackets, are used to represent an optional component. This is to avoid confusion with memory operands which are surrounded by square brackets.

Also, [immea](Opcode#imme.md) and [composite operand](SourceFormat#Composite_operand.md) may be mentioned in the parts below. Please visit these two links for more information regarding their meaning.

### Miscellaneous Instructions ###


#### S2R ####
Store special register to general-purpose register.

Instruction usage:
```
S2R reg0, SRName;
```
SRName could be one of the names specified below, or it could be SRxxx, where xxx is a non-negative integer less than 256.

Template opcode:
```
0010 000000 1110 000000 000000 00000000 000000000000000000000000 110100
                   reg0          SRName
```

|SRName value|SRName|SRName value|SRName|
|:-----------|:-----|:-----------|:-----|
|0 |SR\_LaneId|		38|SR\_CTAid\_Y|
|2 |SR\_VirtCfg|		39|SR\_CTAid\_Z|
|3 |SR\_VirtId|		40|SR\_NTid|
|4 |SR\_PM0|			41|SR\_NTid\_X|
|5 |SR\_PM1|			42|SR\_NTid\_Y|
|6 |SR\_PM2|			43|SR\_NTid\_Z|
|7 |SR\_PM3|			44|SR\_GridParam|
|8 |SR\_PM4|			45|SR\_NCTAid\_X|
|9 |SR\_PM5|			46|SR\_NCTAid\_Y|
|10|SR\_PM6|			47|SR\_NCTAid\_Z|
|11|SR\_PM7|			48|SR\_SWinLo|
|16|SR\_PRIM\_TYPE|	49|SR\_SWINSZ|
|17|SR\_INVOCATION\_ID|50|SR\_SMemSz|
|18|SR\_Y\_DIRECTION|	51|SR\_SMemBanks|
|24|SR\_MACHINE\_ID\_0|	52|SR\_LWinLo|
|25|SR\_MACHINE\_ID\_1|	53|SR\_LWINSZ|
|26|SR\_MACHINE\_ID\_2|	54|SR\_LMemLoSz|
|27|SR\_MACHINE\_ID\_3|	55|SR\_LMemHiOff|
|28|SR\_AFFINITY|		56|SR\_EqMask|
|32|SR\_Tid|			57|SR\_LtMask|
|33|SR\_Tid\_X|		58|SR\_LeMask|
|34|SR\_Tid\_Y|		59|SR\_GtMask|
|35|SR\_Tid\_Z|		60|SR\_GeMask|
|36|SR\_CTAParam|		80|SR\_ClockLo|
|37|SR\_CTAid\_X|		81|SR\_ClockHi|

Related dimensions:
```
c [0x0] [0x8] : %ntid.x
c [0x0] [0xc] : %ntid.y
c [0x0] [0x10]: %ntid.z
c [0x0] [0x14]: %nctaid.x
c [0x0] [0x18]: %nctaid.y
c [0x0] [0x1c]: %nctaid.z
BFE VirtId, 0x914 : %smid
BFE VirtCfg, 0x914: %nsmid
```


---

#### LEPC ####
Instructon usage:
```
LEPC reg0;
```
Template opcode
```
0010 000000 1110 000000 000000 00000000000000000000000000000000 100010
                   reg0
```

---

#### CCTL ####
Instruction usage:
```
CCTL(.E)(.Op1).Op2 reg0, [reg1+0xabcd];
```
0xabcd should be a multiple of 4.
Template opcode:
```
1010 000000 1110 000000 000000   00 000000000000000000000000000000    0 11001
        mod        reg0   reg1 mod2                         0xabcd mod3
```
|mod 2:4 value|.Op2   |
|:------------|:------|
|0            |QRY1   |
|1            |PF1    |
|2            |PF1\_5  |
|3            |PR2    |
|4            |WB     |
|5            |IV     |
|6            |IVALL  |
|7            |RS     |

|mod2 value|.Op1   |
|:---------|:------|
|0         |default|
|1         |.U     |
|2         |.C     |
|3         |.I     |

|mod3|meaning|
|:---|:------|
|0   |default|
|1   |.E     |


---

#### CCTLL ####
Instruction usage:
```
CCTLL.Op1 reg0, [reg1 + 0xabcd];
```
0xabcd should can be at most 24 bits long and should be a multiple of 4. asfermi will allow numbers that are not multiples of 4 to be processed and written in the opcodes, but cuobjdump ignores the lowest 2 bits, and the hardware's behaviour is not confirmed.

Template opcode
```
1010 000000 1110 000000 000000 000000000000000000000000 00000000 001011
        mod        reg0   reg1                   0xabcd 
```

|mod 2:4 value|.Op2   |
|:------------|:------|
|0            |QRY1   |
|1            |PF1    |
|2            |PF1\_5  |
|3            |PR2    |
|4            |WB     |
|5            |IV     |
|6            |IVALL  |
|7            |RS     |


---

#### PSETP ####
Instruction usage:
```
PSETP.(Mainop)(.Logicop) p0, p1, (!)p2, (!)p3, ((!)p4);
```
Template opcode
```
0010 000000 1110 000 000 1110 00 0000   00 00000000000000000 0000 00000 110000
                  p1  p0   p4      p3 mod2                     p2  mod3
```

|mod2 0:1|.Mainop|
|:-------|:------|
|00      |.AND   |
|10      |.OR    |
|01      |.XOR   |
|11      |invalid|

|mod3 3:4|.Logicop|
|:-------|:-------|
|00      |.AND   |
|10      |.OR    |
|01      |.XOR   |
|11      |invalid|