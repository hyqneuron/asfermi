# Instructions #
## Currently Supported ##
  * MOV, MOV32I, LD, LDU, LDL, LDC, LDS, ST, STL, STS, LDLK, LDSLK, STUL, STSUL
  * FADD, FADD32I, FMUL, FMUL32I, FFMA, FSETP, FCMP, MUFU, DADD, DMUL, DFMA, DSETP
  * IADD, IADD32I, IMUL, IMUL32I, IMAD, ISCADD, ISETP, ICMP
  * I2F, I2I, F2I, F2F
  * LOP, LOP32I, SHL, SHR, BFE, BFI, SEL
  * SSY, BRA, JMP, JCAL, CAL, PRET, RET, EXIT, PBK, BRK, PCNT, CONT, PLONGJMP, LONGJMP, NOP, BAR, B2R, ATOM, RED, VOTE
  * S2R, LEPC, CCTL, CCTLL, PSETP

For more information regarding the supported instructions, their usage and opcodes, please see the [Opcode pages](IndexPage#Fermi_ISA.md).

## All Known Instructions ##
The following part of this page lists all instruction names that are currently known. For instructions and their corresponding opcode, see [nanb](nanb.md) and [Opcode](Opcode.md).

For better explanation of those instructions, please refer to the cuobjdump.pdf that comes with CUDA toolkit 4.0


## In Alphabetical Order ##
Instructions that end with _N are 4-byte instructions._

### A-C ###
  * ALD
  * ALD\_N
  * AST
  * AST\_N
  * ATOM
  * B2R
  * BFE
  * BFI
  * BPT
  * BRA
  * BRA\_N
  * BRK
  * BRX
  * CAL
  * CCTL
  * CCTLL
  * CONT
  * CSET
  * CSETP

### D-F ###
  * DADD
  * DFMA
  * DMNMX
  * DMUL
  * DSET
  * DSETP
  * EXIT
  * F2I
  * FADD
  * FADD\_N
  * FCCO
  * FCMP
  * FFMA
  * FFMA\_N
  * FFMA32I
  * FLO
  * FMNMX
  * FMUL
  * FMUL32I
  * FSET
  * FSETP
  * FSETP\_N
  * FSWZ

### I-K ###
  * I2F
  * I2I
  * I2I\_N
  * IADD
  * IADD\_N
  * IADD32I
  * ICMP
  * IMAD
  * IMAD\_N
  * IMAD32I
  * IMNMX
  * IMUL
  * IMUL\_N
  * IMUL32I
  * IPA
  * IPA\_N
  * ISAD
  * ISCADD
  * ISCADD32I
  * ISET
  * ISETP
  * ISETP\_N
  * JCAL
  * JMP
  * JMX
  * KIL

### L-N ###
  * LD
  * LD\_LDU
  * LD\_N
  * LDC
  * LDC\_N
  * LDL
  * LDL\_N
  * LDLK
  * LDS\_LDU
  * LDS\_N
  * LDSLK
  * LDU
  * LEPC
  * LONGJMP
  * LOP
  * LOP\_N
  * LOP32I
  * MEMBAR
  * MOV
  * MOV\_N
  * MOV32I
  * MUFU
  * NOP
  * NOP\_N

### O-R ###
  * OUT
  * P2R
  * PBK
  * PCNT
  * PIXLD
  * PLONGJMP
  * POPC
  * PRET
  * PRMT
  * PSET
  * PSETP
  * R2P
  * RAM
  * RED
  * RET
  * RTT

### S-T ###
  * S2R
  * SAM
  * SEL
  * SHL
  * SHR
  * SSY
  * ST
  * ST\_N
  * STL
  * STL\_N
  * STP
  * STS\_N
  * STSUL
  * STUL
  * SULD
  * SULEA
  * SUQ
  * SURED
  * SUST
  * TEX
  * TLD
  * TLD4
  * TMML
  * TXA
  * TXD
  * TXQ

### V ###
  * VABSDIFF
  * VABSDIFF2
  * VABSDIFF4
  * VADD
  * VADD2
  * VADD4
  * VILD
  * VMAD
  * VMAD\_N
  * VMNMX
  * VMNMX2
  * VMNMX4
  * VOTE
  * VSEL2
  * VSEL4
  * VSET
  * VSET2
  * VSET4
  * VSETP
  * VSHL
  * VSHL2
  * VSHL4
  * VSHR
  * VSHR2
  * VSHR4

## By Type ##
Note: the following names are taken from cuobjdump.pdf and are inaccurate.
### Floating point Instructions ###
  * FFMA
  * FADD
  * FCMP
  * FMUL
  * FMNMX
  * FSWZ
  * FSET
  * FSETP
  * RRO
  * MUFU
  * DFMA
  * DADD
  * DMUL
  * DMNMX
  * DSET
  * DSETP

### Integer Instructions ###
  * IMAD
  * IMUL
  * IADD
  * ISCADD
  * ISAD
  * ISAD
  * IMNMX
  * BFE
  * BFI
  * SHR
  * SHL
  * LOP
  * FLO
  * ISET
  * ISETP
  * ICMP
  * POPC

### Conversion Instructions ###
  * F2F
  * F2I
  * I2F
  * I2I

### Movement Instructions ###
  * MOV
  * SEL
  * PRMT

### Predicate/CC Instructions ###
  * P2R
  * R2P
  * CSET
  * CSETP
  * PSET
  * PSETP

### Texture Instructions ###
  * TEX
  * TLD
  * TLD4
  * TXO

### LD/ST Instructions ###
  * LDC
  * LD
  * LDU
  * LDL
  * LDS
  * LDLK
  * LDSLK
  * LD\_LDU
  * LDS\_LDU
  * ST
  * STL
  * STUL
  * STS
  * STSUL
  * ATOM
  * RED
  * CCTL
  * CCTLL
  * MEMBAR

### Surface Memory Instructions ###
  * SULD
  * SULEA
  * SUST
  * SURED
  * SUQ

### Control Instructions ###
  * BRA
  * BRX
  * JMP
  * JMX
  * CAL
  * JCAL
  * RET
  * BRK
  * CONT
  * LONGJMP
  * SSY
  * PBK
  * PCNT
  * PRET
  * PLONGJMP
  * BPT
  * EXIT

### Miscellaneous Instructions ###
  * NOP
  * S2R
  * B2R
  * LEPC
  * BAR
  * VOTE