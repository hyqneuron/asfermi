This page is written by hyq.neuron for the other asfermi team members.



## Probing opcode ##
  1. Familiarize yourself with the bit fields of Fermi's opcodes. You can look through all the [Opcode](Opcode.md) pages to get a rough idea.
  1. Choose an instruction you intend to probe
  1. Find out the possible corresponding PTX instruction
  1. Code in PTX (here's a template file I often use: [PTXHelp](PTXHelp.md)), with as many permutations of suitable modifiers as possible, and produce the corresponding cubin. PTX modifiers can be found from the PTX manual 2.3 from CUDA Toolkit 4.0
  1. Use cuobjdump to disassemble the cubin. From the disassembled code, identify the types of operands for this instruction
  1. Use cubinEditor from [Downloads](http://code.google.com/p/asfermi/downloads/list) to [open](Utilities#cubin_binary_editor.md) the cubin
  1. Document the template opcode on the correct opcode page (We'll discuss which category an instruction belongs to before we start probing). The default modifier bits should be present, so should the default predicate bits(1110). However, please leave all other operand-specific bits as zeros.
  1. Apart from the variation in modifiers that are already present, please also check for other potential modifier bits by tweaking the unused bit fields.
  1. Identify and document the meaning of various modifier bit fields. If you do not understand what a certain modifier means, please leave a question mark at the end of its name, e.g. .H?.

Things will be clearer after you have thoroughly looked through the opcode pages that I have written so far.

An example of doucmentation taken from [OpcodeFloat](OpcodeFloat.md):

---

Instruction usage:
```
FMUL(.rnd)(.SAT) reg0, reg1, composite operand;
```
Template opcode:
```
0000 000000 1110 000000 000000 0000000000000000000000 0000000000 011010
        mod        reg0   reg1                  immea       mod2
```
|mod2 1:2 | .rnd|
|:--------|:----|
|00|default (.rn) |
|10|.RM |
|01|.RP |
|11|.RZ |

|mod 4 value|meaning|
|:----------|:------|
|0 |default|
|1 |.SAT|


---

A few more examples: [OpcodeConversion#F2I](OpcodeConversion#F2I.md), [OpcodeFloat#FSETP](OpcodeFloat#FSETP.md)


---

## Adding support for instructions in asfermi ##
`This is where you'll have to start coding. A few structures must be introduced: SubString(SubString.h), Instruction(DataTypes.h), ModifierRule, OperandRule, InstructionRule(DataTypes.h).`

`*SubString*: SubString structure contains a char pointer (Start) to the beginning of a sub-section of a char array, as well as an integer (Length) indicating the length of the sub-string.`

`You can read the comments in SubString.h first. There is no need to go into the parsing functions in SubString.h for now. Then you can proceed to reading DataTypes.h to understand the other structures. While you are reading DataTypes.h, it's better that you jump to some examples from time to time. For instance, while reading the part of the  ModifierRule structure, you can go to RulesModifierFloat.h to see how ModifierRule instances can be implemented. I've added more comments in RulesOperandRegister.h RulesInstructionLogic.h and RulesInstructionDataMovement.h to illustrate things.`

Use VS's feature right click -> Go To Definition for any function/structure that you do not understand.

I believe the rules are simple enough for you to generate easily. The only problem I see is with the operand rules. It is worth noting that there are basically 2 types of operand rules: simple and composite. Simple operands are simple in the sense that they are of fixed types. Composite operand can be constant memory, 20-bit constant or register.

Some operands take operators such as '~', '-' or '||'. The general purpose operand rules do not process such operators as they are instruction-specific. As a result, to enable the processing of such operators separate operand rules must be written. See OPRFADD32IReg1 (a simple operand rule) in RulesOperandRegister.h for an example. Those operators are often applicable to arithmetic/logic instructions, which usually accept composite operands. Here's an example of a composite operand rule (the actual code in RulesOperandComposite.h is harder to read due to the use of macros)`
```
struct OperandRuleFADDStyle: OperandRule
{
	bool AllowNegative;
	OperandRuleFADDStyle(bool allowNegative) :OperandRule(FADDStyle)
	{
		AllowNegative = allowNegative;
	}
	virtual void Process(SubString &component)
	{
		//Register or constant memory

		bool negative = false;
		if(component[0] == '-')
		{
			if(!AllowNegative)
				throw 129;
			negative = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<8; 
			component.Start++;
			component.Length--;
			if(component.Length<1) 
				throw 116; 
		}
		/*register*/
		if(component[0]=='R' || component[0]=='r')
		{
			int register2 = component.ToRegister();
			csMaxReg = (register2 > csMaxReg)? register2: csMaxReg;
			csCurrentInstruction.OpcodeWord0 |= register2 << 26;
			MarkRegisterForImmediate32();
		}		
		/*constant memory*/
		else if(component[0]=='c' || component[0] == 'C') 
		{
			unsigned int bank, memory;
			int register1;
			component.ToConstantMemory(bank, register1, memory);
			if(register1 != 63)
				throw 112;
			csCurrentInstruction.OpcodeWord1 |= bank<<10;
			WriteToImmediate32(memory);
			MarkConstantMemoryForImmediate32();
		}
		//constants
		else
		{
			unsigned int result;
			//float
			if(component[0]=='F')
				result = component.ToImmediate20FromFloatConstant();
			//hex
			else if(component.Length>2 && component[0]=='0' &&(component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate20FromHexConstant(true);
			else
				throw 116;
			
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		if(negative)
		{
			component.Start--;
			component.Length++;
		}
	}
}OPRFADDStyle(true), OPRFMULStyle(false);
```

Constant operads:
  * OPRImmediate24HexConstant
  * OPRImmediate32HexConstant
  * OPRImmediate32IntConstant
  * OPRImmediate32FloatConstant
  * OPRS2R //Special register names for S2R only
Note that the hex operand rules accept negative hex expressions.

Register operands:
  * OPRRegister0 //reg0
  * OPRRegister1 //reg1
  * OPRRegister2 //reg2
  * OPRRegister3 //reg3
  * OPRRegisterWithCC4IADD32I
  * OPRPredicate1 //doesn't process !
  * OPRPredicate0 //doesn't process !
  * OPRPredicate2NotNegatable //doesn't process !
  * OPRPredicate2 //processes !
  * OPRFADD32IReg1 //for FADD32I, processes -.

Memory operands:
  * OPRGlobalMemoryWithImmediate32
  * OPRSharedMemoryWithImmediate20
  * OPRConstantMemory

Composite operands:
  * OPRMOVStyle
  * OPRFADDStyle (processes -. doesn't process int expressions)
  * OPRFMULStyle (doesn't process -)
  * OPRIADDStyle (processes -. doesn't process float exp)
  * OPRIMULStyle (doesn't process -)

Other operands:
  * OPR32I (for all 32I operands)
  * OPRLOP1
  * OPRLOP2
  * OPRF2I
  * OPRI2F

You may look through the operand rules before you start writing your own operand rule. Some operand rules can be reused.

After all the rules have been created, add the instruction rule to Initialize() in asfermi.cpp and asfermi should then be able to parse the instruction.


---

## Probing architectural features ##