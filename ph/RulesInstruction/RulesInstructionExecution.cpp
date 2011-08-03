#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"

#include "stdafx.h"

#include "RulesInstructionExecution.h"
#include "..\RulesModifier.h"
#include "..\RulesOperand.h"


struct InstructionRuleEXIT: InstructionRule
{
	InstructionRuleEXIT() : InstructionRule("EXIT", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IREXIT;

struct InstructionRuleCAL: InstructionRule
{
	InstructionRuleCAL() : InstructionRule("CAL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000000000100000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate24HexConstant);
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRCAL;

struct InstructionRuleBRA: InstructionRule
{
	InstructionRuleBRA() : InstructionRule("BRA", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate24HexConstant);
		ModifierGroups[0].Initialize(true, 1, &MRBRAU);
	}
}IRBRA;

struct InstructionRulePRET: InstructionRule
{
	InstructionRulePRET() : InstructionRule("PRET", 1, true, false)
	{
		hpBinaryStringToOpcode8("1110 000000000001000000000000000000000000000000000000000000011110", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate24HexConstant);
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRPRET;



struct InstructionRuleRET: InstructionRule
{
	InstructionRuleRET() : InstructionRule("RET", 0, true, false)
	{
		hpBinaryStringToOpcode8("1110 011110111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
	}
}IRRET;


struct InstructionRuleNOP: InstructionRule
{
	InstructionRuleNOP(): InstructionRule("NOP", 3, true, false)
	{
		hpBinaryStringToOpcode8("0010 011110 111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRNOPCC, &OPRImmediate16HexOrIntOptional);
		ModifierGroups[0].Initialize(true, 1, &MRNOPTRIG);
		ModifierGroups[1].Initialize(true, 8, 
										&MRNOPFMA64,
										&MRNOPFMA32,
										&MRNOPXLU  ,
										&MRNOPALU  ,
										&MRNOPAGU  ,
										&MRNOPSU   ,
										&MRNOPFU   ,
										&MRNOPFMUL);
		ModifierGroups[2].Initialize(true, 1, &MRS);
	}
}IRNOP;