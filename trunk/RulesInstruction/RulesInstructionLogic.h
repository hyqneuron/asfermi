#ifndef RulesInstructionLogicDefined


struct InstructionRuleLOP: InstructionRule
{
	InstructionRuleLOP(): InstructionRule("LOP", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010110", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRLOP1,
					&OPRLOP2);
		ModifierGroups[0].Initialize(false, 4,
					&MRLOPAND,
					&MRLOPOR,
					&MRLOPXOR,
					&MRLOPPASS);
	}
}IRLOP;



#else
#define RulesInstructionLogicDefined
#endif