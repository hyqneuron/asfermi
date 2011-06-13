#if defined SpecificRulesDefined //prevent multiple inclusion
#else
#define SpecificRulesDefined yes
//---code starts ---
#include <vld.h> //remove when you compile


#include <iostream>
#include <list>
#include "DataTypes.h"

extern void hpPrintBinary8(unsigned int word0, unsigned int word1);

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
struct OperandRuleRegister: OperandRule
{
	int Offset;
	OperandRuleRegister()
	{
		ModifierCount = 0;
		Type = (OperandType)0;
	}
	OperandRuleRegister(int offset)
	{
		ModifierCount = 0;
		Type = (OperandType)0;
		Offset = offset;
	}
	virtual void Process(Instruction &instruction, Component &component)
	{
		int result = component.Content.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		result = result<<Offset;
		instruction.OpcodeWord0 |= result;
	}
};
OperandRuleRegister OPRRegister0(14), OPRRegister1(20), OPRRegister2(26);

struct OperandRuleImmediate32: OperandRule
{
	OperandRuleImmediate32()
	{
		ModifierCount = 0;
		Type = (OperandType)1;
	}
	virtual void Process(Instruction &instruction, Component &component)
	{
		//only works for hex value
		unsigned int result = component.Content.ToImmediate32FromHex();
		instruction.OpcodeWord0 |= result<<26;
		instruction.OpcodeWord1 |= result>>6;
	}	
}OPRImmediate32;
struct OperandRuleGlobalMemoryWithImmediate32: OperandRule
{
	OperandRuleGlobalMemoryWithImmediate32()
	{
		ModifierCount = 0;
		Type = (OperandType)2;
	}
	virtual void Process(Instruction &instruction, Component &component)
	{
		unsigned int memory; int register1;
		component.Content.ToGlobalMemory(register1, memory);
		csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;
		instruction.OpcodeWord0 |= register1<<20; //RE1
		instruction.OpcodeWord0 |= memory << 26;
		instruction.OpcodeWord1 |= memory >> 6;
	}
}OPRGlobalMemoryWith32Immediate;
struct OperandRuleMOVStyle: OperandRule
{
	OperandRuleMOVStyle()
	{
		ModifierCount = 0;
		Type = MOVStyle;
	}
	virtual void Process(Instruction &instruction, Component &component)
	{
		if(component.Content[0]=='c'||component.Content[0]=='C')
		{
			unsigned int bank, memory;
			int register1;
			component.Content.ToConstantMemory(bank, register1, memory);
			if(register1 != 63)
				throw 112; //register cannot be used in MOV-style constant address
			instruction.OpcodeWord1 |= bank<<10;
			instruction.OpcodeWord0 |= memory << 26;
			instruction.OpcodeWord1 |= memory >> 6;
			instruction.OpcodeWord1 |= 1<<14;
		}
		else if(component.Content[0] == 'r' || component.Content[0] == 'R')
		{
			OPRRegister2.Process(instruction, component);
			//::hpPrintBinary8(instruction.OpcodeWord0, instruction.OpcodeWord1);
		}
		else
		{
			unsigned int memory = component.Content.ToImmediate32FromHex();
			if(memory>0xFFFF)
				throw 113; //The immediate value is limited to 16-bit.
			instruction.OpcodeWord0 |= memory << 26;
			instruction.OpcodeWord1 |= memory >> 6;
			instruction.OpcodeWord1 |= 3<<14;
		}
	}
}OPRMOVStyle;
struct OperandRuleFADDStyle: OperandRule
{
	OperandRuleFADDStyle()
	{
		ModifierCount = 0;
		Type = FADDStyle;
	}
	virtual void Process(Instruction &instruction, Component &component)
	{
		if(component.Content[0]=='R' || component.Content[0]=='r')
		{
			OPRRegister2.Process(instruction, component);
		}
		else
		{
			unsigned int memory = component.Content.ToImmediate32FromHex();
			if(memory>0xFFFFF)
				throw 115; //limited to 20-bit;
			instruction.OpcodeWord0 |= memory << 26;
			instruction.OpcodeWord1 |= memory >> 6;
			instruction.OpcodeWord1 |= 3<<14;
		}
	}
}OPRFADDStyle;
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
		Operands[1] = &OPRGlobalMemoryWith32Immediate;
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
		Operands[0] = &OPRGlobalMemoryWith32Immediate;
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
		Operands[2] = &OPRFADDStyle;
	}
}IRIADD;
//-----End of specific instruction rules

//	7
//------Specific Modifier Rules
//------End of specific modifier rules

#endif