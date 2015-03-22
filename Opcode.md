Note: information here is not organised. Many parts may be unclear and confusing. Information may be inaccurate.



# Opcode Information #
Currently 2 types of opcodes found: 4-byte and 8-byte. For instruction names and their identifier bits (na, nb), see [Instructions](Instructions.md), [nanb](nanb.md).


---

## 8-byte ##
General format:
```
LSB                                                                MSB
xxx0 xxxxxx xxxx xxxxxx xxxxxx xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx xxxxxx
  na    mod   pr    re0    re1                             imme     nb
```
The numbering of bit fields below, such as re0 3:5, start from the MSB, but the displayed binary digits start from the LSB. For instance, when pr is 0101, then "pr 2" will be 1 and "pr 0:2" will be 101.

### na, nb ###
These two bit fields specify the instruction name. Invalid bits result in invalid opcode.

### pr ###
Predicate bits
|Field|Meaning                           |
|:----|:---------------------------------|
|0    |Negate predicate. Eg: 1001: @!P1  |
|1:3  |Predicate register number. Eg: 0100: @P2|

Note: Predicate register 7 , P7, is actually recognised by cuobjdump as "pt", which is similar to "RZ" in the sense that it possibly means "no predicate" (1110 means no predicate used). Also, P4, P5 and P6 have not been tested. Use of those predicate registers may trigger error.

### re0 ###
re0 is most often used for as the destination register. However, for SETP instructions re0 0:2 is the destination predicate register(p), and re0 3:5 is destination predicate for NOT p(~p).
### re1 ###
re1 is often the second register operand. It is also used to store the Rxx in `[Rxx+0xaabb]` for memory operands.

### imme ###
For instructions with the suffix I32, the entirety of the field is used to store a 32-bit immediate value.

For most other instructions that use this field, imme is divided into two fields: immea(imme10:31) and immeb(imme0:9). immea can be used to indicate a 20-bit immediate value, a constant memory address, or a register. immea is most often used by a [composite operand](SourceFormat#Composite_operand.md)
|immea 0:1|immea 2:21               |
|:--------|:------------------------|
|00        |lowest 6 bits: register |
|10        |constant memory address |
|01        |see note below          |
|11        |20-bit immediate        |

  * When immea 0:1 is 10, imme 2:5 is the constant memory bank number and imme 6:21 is the address within this bank.
  * When immea 0:1 is 01, the instruction should be one that resembles IMAD/FFMA/DFMA, i.e. supporting 4 operands of the same length. The third operand ought to be a register whose register number is stored in the reg3 field. The fourth operand should be a constant memory address stored in the lower 20 bits of immea.

immeb is sometimes used to store modifier bits.


---

## 4-byte ##
General format:
```
LSB                               MSB
xxx1xxx   x     xx xxxx xxxxxx xxxxxx xxxxxx
     na mod  immeb   pr    re1    re0  immea
```