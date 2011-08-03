
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"

#include "stdafx.h"

#include "RulesInstructionMiscellaneous.h"
#include "..\RulesModifier.h"
#include "..\RulesOperand.h"




struct InstructionRuleS2R: InstructionRule
{
	InstructionRuleS2R(): InstructionRule("S2R", 0, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 111000000000000000000000000000000000000000000000110100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRRegister0, &OPRS2R);
	}
}IRS2R;
