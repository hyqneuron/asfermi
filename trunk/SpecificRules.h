#if defined SpecificRulesDefined //prevent multiple inclusion
#else
#define SpecificRulesDefined yes
//---code starts ---
#include <vld.h>


#include <iostream>
#include <list>
#include "DataTypes.h"


//	5
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
OperandRuleRegister OPRRegister0(14), OPRRegister1(20), OPRRegister(26);

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
		instruction.OpcodeWord0 |= register1<<20; //RE1
		instruction.OpcodeWord0 |= memory << 26;
		instruction.OpcodeWord1 |= memory >> 6;
	}
}OPRGlobalMemoryWith32Immediate;
//-----End of Specifc Operand Rules


//	6
//-----Specific Instruction Rules
struct InstructionRuleLD: InstructionRule
{
	InstructionRuleLD() : InstructionRule("LD", 2, 0, true)
	{
		Operands[0] = &OPRRegister0;
		Operands[1] = &OPRGlobalMemoryWith32Immediate;
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IRLD;

struct InstructionRuleST: InstructionRule
{
	InstructionRuleST() : InstructionRule("ST", 2, 0, true)
	{
		Operands[0] = &OPRGlobalMemoryWith32Immediate;
		Operands[1] = &OPRRegister0;
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
	}
}IRST;
struct InstructionRuleEXIT: InstructionRule
{
	InstructionRuleEXIT() : InstructionRule("EXIT", 0, 0, true)
	{
		InstructionRule::BinaryStringToOpcode8("1110011110111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IREXIT;

//-----End of specific instruction rules

//	7
//------Specific Modifier Rules
//------End of specific modifier rules

#endif