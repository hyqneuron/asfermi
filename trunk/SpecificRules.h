/*
This file contains the specific rules for various modifiers, operands and instructions
*/

#ifndef SpecificRulesDefined //prevent multiple inclusion
//---code starts ---


inline void WriteToImmediate32(unsigned int content)
{
	csCurrentInstruction.OpcodeWord0 |= content <<26;
	csCurrentInstruction.OpcodeWord1 |= content >> 6;
}
inline void MarkConstantMemoryForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 1<<14; //constant memory flag
}
inline void MarkImmediate20ForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 3<<14; //20-bit immediate flag
}
inline void MarkRegisterForImmediate32()
{
}


//	5.1
//-----Specific modifier rules
struct ModifierRule128: ModifierRule
{
	ModifierRule128(): ModifierRule("128", 3, true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111001111111111111111111111111", Mask0);
		::InstructionRule::BinaryStringToOpcode4("00000010000000000000000000000000", Bits0);
	}
}MR128;
struct ModifierRule64: ModifierRule
{
	ModifierRule64(): ModifierRule("64", 2, true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111001111111111111111111111111", Mask0);
		::InstructionRule::BinaryStringToOpcode4("00000100000000000000000000000000", Bits0);
	}
}MR64;
//-----End of specific modifier rules









//	5.2
//-----Specific Operand Rules
/*
Primary operand types: 
	Register, Constant, Global Memory, Constant Memory, HexConstant, IntegerConstant, FloatConstant, Predicate, S2R operand
Primary operand types have their range checking done in SubString member functions

Composite operand types:
	MOVStyle(Constant memory without reg, 20-bit Hex)
	FADDStyle(Constant memory without reg, 20-bit float)
	FIADDStyle(Constant memory without reg, 20-bit Int)

Operator processors only check that components have the correct initiator and then send components to SubString
	member functions for processing. Fundamental operator processors then directly write back the result.
Composite operator processor would sometimes have to check the value returned by SubString functions fall within
	a defined, smaller range before writing.
*/
//SubString member functions only assume a least length of 1


//---Constant Operands
struct OperandRuleImmediate32HexConstant: OperandRule
{
	OperandRuleImmediate32HexConstant() :OperandRule(Immediate32HexConstant, 0){}
	virtual void Process(Component &component)
	{
		unsigned int result = component.Content.ToImmediate32FromHexConstant();
		WriteToImmediate32(result);
	}	
}OPRImmediate32HexConstant;

struct OperandRuleImmediate32IntConstant: OperandRule
{
	OperandRuleImmediate32IntConstant():OperandRule(Immediate32IntConstant, 0){}
	virtual void Process(Component &component)
	{
		unsigned int result = component.Content.ToImmediate32FromIntConstant();
		WriteToImmediate32(result);
	}	
}OPRImmediate32IntConstant;

struct OperandRuleImmediate32FloatConstant: OperandRule
{
	OperandRuleImmediate32FloatConstant():OperandRule(Immediate32FloatConstant, 0){}
	virtual void Process(Component &component)
	{
		int modLength = 0;
		if(component.Modifiers.size()!=0)
		{
			modLength = component.Modifiers.begin()->Length;
			component.Modifiers.pop_front();
		}
		unsigned int result = component.Content.ToImmediate32FromFloatConstant(modLength);
		WriteToImmediate32(result);
	}	
}OPRImmediate32FloatConstant;

struct OperandRuleImmediate32AnyConstant: OperandRule
{
	OperandRuleImmediate32AnyConstant():OperandRule(Immediate32AnyConstant, 0){}
	virtual void Process(Component &component)
	{
		//Issue: not yet implemented
	}	
}OPRImmediate32AnyConstant;
//---End of constant operands








//---Core operands
struct OperandRuleRegister: OperandRule
{
	int Offset;
	OperandRuleRegister(): OperandRule(Register, 0){}
	OperandRuleRegister(int offset)
	{
		ModifierCount = 0;
		Type = (OperandType)0;
		Offset = offset;
	}
	virtual void Process(Component &component)
	{
		int result = component.Content.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord0 |= result;
	}
};
OperandRuleRegister OPRRegister0(14), OPRRegister1(20), OPRRegister2(26);

struct OperandRuleGlobalMemoryWithImmediate32: OperandRule
{
	OperandRuleGlobalMemoryWithImmediate32(): OperandRule(GlobalMemoryWithImmediate32, 0){}
	virtual void Process(Component &component)
	{
		unsigned int memory; int register1;
		component.Content.ToGlobalMemory(register1, memory);
		csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		WriteToImmediate32(memory);
	}
}OPRGlobalMemoryWithImmediate32;

struct OperandRuleConstantMemory: OperandRule
{
	OperandRuleConstantMemory() : OperandRule(ConstantMemory, 0){}
	virtual void Process(Component &component)
	{		
		unsigned int bank, memory;
		int register1;
		component.Content.ToConstantMemory(bank, register1, memory);
		csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		csCurrentInstruction.OpcodeWord1 |= bank<<10;
		WriteToImmediate32(memory);
		//no need to do the marking for constant memory
	}
}OPRConstantMemory;

struct OperandRuleIgnored: OperandRule
{
	OperandRuleIgnored() : OperandRule(OperandType::Optional, 0){}
	virtual void Process(Component &component)
	{
		//do nothing
	}
}OPRIgnored;
//---End of core operands








//---Instruction-specific operands
struct OperandRuleMOVStyle: OperandRule
{
	OperandRuleMOVStyle() : OperandRule(MOVStyle, 0){}

	virtual void Process(Component &component)
	{		
		//seems MOV doesn't output -
		//bool negate = false;
		//if(component.Content[0] == '-')
		//{
		//	negate = true;
		//	component.Content.Start++;
		//	component.Content.Length--;
		//	if(component.Content.Length<3 || component.Content[0]!='0' || (component.Content[1]!='x'&&component.Content[1]!='X') )
		//		throw 116; //invalid operand
		//}

		if(component.Content[0] == 'R' || component.Content[0] == 'r')
			OPRRegister2.Process(component);
		else if(component.Content[0]=='c'||component.Content[0]=='C')
		{
			unsigned int bank, memory;
			int register1;
			component.Content.ToConstantMemory(bank, register1, memory);
			if(register1 != 63)
				throw 112; //register cannot be used in MOV-style constant address
			csCurrentInstruction.OpcodeWord1 |= bank<<10;
			WriteToImmediate32(memory);
			MarkConstantMemoryForImmediate32();
		}
		else //for MOV, only hex constants are allowed
		{
			unsigned int result = component.Content.ToImmediate20FromHexConstant();
			/*if(negate)
			{
				result ^= 0xFFFFF;
				result += 1;
				result &= 0xFFFFF;
				component.Content.Start--;
				component.Content.Length++;
			}*/
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
	}
}OPRMOVStyle;

//issue: will not work if the second operand, a register, has a negative sign before it. This is, however, possible
//since it's been observed in the output of cuobjdump. The mod1:0 bits, when = 10, negate third operand; when = 01, negate second operand;
//when = 11, produces.PO, which I do not understand for now
struct OperandRuleFADDStyle: OperandRule
{
	OperandRuleFADDStyle() :OperandRule(FADDStyle, 0){}
	virtual void Process(Component &component)
	{
		bool negate = false;
		if(component.Content[0] == '-')
		{
			negate = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<8; //negate modifier bit
			component.Content.Start++;
			component.Content.Length--;
			if(component.Content.Length<1) //R0, c[xx][xx], 0, still allows the SubString functions to assume length >=1
				throw 116; //invalid operand
		}

		if(component.Content[0]=='R' || component.Content[0]=='r')
		{
			int register2 = component.Content.ToRegister();
			csMaxReg = (register2 > csMaxReg)? register2: csMaxReg;
			csCurrentInstruction.OpcodeWord0 |= register2 << 20; //RE2
			MarkRegisterForImmediate32();
		}		
		else if(component.Content[0]=='c' || component.Content[0] == 'C') //constant memory
		{
			unsigned int bank, memory;
			int register1;
			component.Content.ToConstantMemory(bank, register1, memory);
			if(register1 != 63)
				throw 112; //register cannot be used in FADD-style constant address
			csCurrentInstruction.OpcodeWord1 |= bank<<10;
			WriteToImmediate32(memory);
			MarkConstantMemoryForImmediate32();
		}
		else if(component.Content[0]=='F') //float constant
		{
			int modLength = 0;
			if(component.Modifiers.size()!=0)
			{
				int modLength = component.Modifiers.begin()->Length;
				component.Modifiers.pop_front();
			}
			unsigned int result = component.Content.ToImmediate20FromFloatConstant(modLength);
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		else if(component.Content.Length>2 && component.Content[0]=='0' &&(component.Content[1]=='x'||component.Content[1]=='X'))
		{
			unsigned int result = component.Content.ToImmediate20FromHexConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		else
			throw 116;
		if(negate)
		{
			component.Content.Start--;
			component.Content.Length++;
		}
	}
}OPRFADDStyle;


struct OperandRuleIADDStyle: OperandRule
{
	OperandRuleIADDStyle() :OperandRule(IADDStyle, 0){}
	virtual void Process(Component &component)
	{
		bool negate = false;
		if(component.Content[0] == '-')
		{
			negate = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<8; //negate modifier bit
			component.Content.Start++;
			component.Content.Length--;
			if(component.Content.Length<1) //R0, c[xx][xx], 0, still allows the SubString functions to assume length >=1
				throw 116; //invalid operand
		}

		if(component.Content[0]=='R' || component.Content[0]=='r')
		{
			int register2 = component.Content.ToRegister();
			csMaxReg = (register2 > csMaxReg)? register2: csMaxReg;
			csCurrentInstruction.OpcodeWord0 |= register2 << 20; //RE2
			MarkRegisterForImmediate32();
		}		
		else if(component.Content[0]=='c' || component.Content[0] == 'C') //constant memory
		{
			unsigned int bank, memory;
			int register1;
			component.Content.ToConstantMemory(bank, register1, memory);
			if(register1 != 63)
				throw 112; //register cannot be used in FADD-style constant address
			csCurrentInstruction.OpcodeWord1 |= bank<<10;
			WriteToImmediate32(memory);
			MarkConstantMemoryForImmediate32();
		}
		else if(component.Content.Length>2 && component.Content[0]=='0' &&(component.Content[1]=='x'||component.Content[1]=='X'))
		{
			unsigned int result = component.Content.ToImmediate20FromHexConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		else //int
		{
			unsigned int result = component.Content.ToImmediate20FromIntConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		if(negate)
		{
			component.Content.Start--;
			component.Content.Length++;
		}
	}
}OPRIADDStyle;
//---End of instruction-specific operands
//-----End of Specifc Operand Rules


//	6
//-----Specific Instruction Rules
struct INstructionRuleMOV: InstructionRule
{
	INstructionRuleMOV(): InstructionRule("MOV", 2, 0, true, false)
	{
		Operands[0] = &OPRRegister0;
		Operands[1] = &OPRMOVStyle;
		InstructionRule::BinaryStringToOpcode8("0010011110111000000000000000000000000000000000000000000000010100", OpcodeWord0, OpcodeWord1);
	}
}IRMOV;
struct InstructionRuleLD: InstructionRule
{
	InstructionRuleLD() : InstructionRule("LD", 2, 2, true, false)
	{
		Operands[0] = &OPRRegister0;
		Operands[1] = &OPRGlobalMemoryWithImmediate32;
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
		ModifierRules[0] = &MR128;
		ModifierRules[1] = &MR64;
	}
}IRLD;

struct InstructionRuleST: InstructionRule
{
	InstructionRuleST() : InstructionRule("ST", 2, 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
		Operands[0] = &OPRGlobalMemoryWithImmediate32;
		Operands[1] = &OPRRegister0;
		ModifierRules[0] = &MR128;
		ModifierRules[1] = &MR64;
	}
}IRST;
struct InstructionRuleEXIT: InstructionRule
{
	InstructionRuleEXIT() : InstructionRule("EXIT", 0, 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110011110111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IREXIT;
struct InstructionRuleFADD: InstructionRule
{
	InstructionRuleFADD() : InstructionRule("FADD", 3, 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0000000000111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		Operands[0] = &OPRRegister0;
		Operands[1] = &OPRRegister1;
		Operands[2] = &OPRFADDStyle;
	}
}IRFADD;
struct InstructionRuleIADD: InstructionRule
{
	InstructionRuleIADD() : InstructionRule("IADD", 3, 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010010", OpcodeWord0, OpcodeWord1);
		Operands[0] = &OPRRegister0;
		Operands[1] = &OPRRegister1;
		Operands[2] = &OPRIADDStyle;
	}
}IRIADD;
struct INstructionRuleNOP: InstructionRule
{
	INstructionRuleNOP(): InstructionRule("NOP",1 , 0, true, false)
	{
		Operands[0] = &OPRIgnored;
		InstructionRule::BinaryStringToOpcode8("0010011110111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
	}
}IRNOP;
//-----End of specific instruction rules

//	7
//------Specific Modifier Rules
//------End of specific modifier rules

#else
#define SpecificRulesDefineds
#endif