#ifndef RulesInstructionLogicDefined


struct InstructionRuleLOP: InstructionRule
{
	//InstructionRule(char* name, int modifierGroupCount, bool is8, bool needCustomProcessing)
	InstructionRuleLOP(): InstructionRule("LOP", 1, true, false)
	{
		//set template opcode
		InstructionRule::BinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010110", OpcodeWord0, OpcodeWord1);
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



#else
#define RulesInstructionLogicDefined
#endif