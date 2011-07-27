
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"

#include "stdafx.h"

#include "RulesInstructionLogic.h"
#include "..\RulesModifier.h"
#include "..\RulesOperand.h"


struct InstructionRuleLOP: InstructionRule
{
	//InstructionRule(char* name, int modifierGroupCount, bool is8, bool needCustomProcessing)
	InstructionRuleLOP(): InstructionRule("LOP", 1, true, false)
	{
		//set template opcode
		hpBinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010110", OpcodeWord0, OpcodeWord1);
		//Set operands
		//SetOperands(int operandCount, OperandRule*...)
		SetOperands(3,
					&OPRRegister0,
					&OPRLOP1,
					&OPRLOP2);
		//The array ModifierGroups is initialized in the constructor. Now each ModifierGroup needs to be initialized once
		//Initialize(int modifierCount, ModifierRule*...);
		ModifierGroups[0].Initialize(false, 4,
					&MRLOPAND,
					&MRLOPOR,
					&MRLOPXOR,
					&MRLOPPASS);
	}
}IRLOP;


struct InstructionRuleSHR: InstructionRule
{
	InstructionRuleSHR(bool shr) : InstructionRule("", 2, true, false)
	{
		if(shr)
		{
			hpBinaryStringToOpcode8("1100 010001 1110 000000 000000 0000000000000000000000 0000000000 011010", OpcodeWord0, OpcodeWord1);
			Name = "SHR";
		}
		else
		{
			hpBinaryStringToOpcode8("1100 010001 1110 000000 000000 0000000000000000000000 0000000000 000110", OpcodeWord0, OpcodeWord1);
			Name = "SHL";
		}
		SetOperands(3,
					&OPRRegisterWithCCAt16, 
					&OPRRegister1,
					&OPRIADDStyle);
		ModifierGroups[0].Initialize(true, 1, &MRSHRU32);
		ModifierGroups[1].Initialize(true, 1, &MRSHRW);
	}
}IRSHR(true), IRSHL(false);