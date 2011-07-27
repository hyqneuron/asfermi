
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"

#include "stdafx.h"

#include "RulesInstructionMiscellaneous.h"
#include "..\RulesModifier.h"
#include "..\RulesOperand.h"


struct InstructionRuleNOP: InstructionRule
{
	InstructionRuleNOP(): InstructionRule("NOP", 0, true, false)
	{
		SetOperands(1, &OPRIgnored);
		hpBinaryStringToOpcode8("0010 011110 111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
	}
}IRNOP;


struct InstructionRuleS2R: InstructionRule
{
	InstructionRuleS2R(): InstructionRule("S2R", 0, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 111000000000000000000000000000000000000000000000110100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRRegister0, &OPRS2R);
	}
}IRS2R;
