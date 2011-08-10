#ifndef RulesInstructionIntegerDefined



struct InstructionRuleIADD: InstructionRule
{
	InstructionRuleIADD() : InstructionRule("IADD", 0, true, false)
	{
		hpBinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRRegister1,
					&OPRIADDStyle);
	}
}IRIADD;


struct INstructionRuleIADD32I: InstructionRule
{
	INstructionRuleIADD32I(): InstructionRule("IADD32I", 2, true, false)
	{
		hpBinaryStringToOpcode8("0100000000111000000000000000000000000000000000000000000000010000", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegisterWithCC4IADD32I,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 1, &MRIADD32ISAT);
		ModifierGroups[1].Initialize(true, 1, &MRIADD32IX);
	}
}IRIADD32I;


struct InstructionRuleIMUL: InstructionRule
{
	InstructionRuleIMUL(): InstructionRule("IMUL", 4, true, false)
	{
		hpBinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRRegister1,
					&OPRIMULStyle);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
		ModifierGroups[3].Initialize(true, 1, &MRIMULSAT);
	}
}IRIMUL;

struct InstructionRuleIMAD: InstructionRule
{
	InstructionRuleIMAD(): InstructionRule("IMAD", 4, true, false)
	{
		hpBinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000000100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister0,
					&OPRRegister1,
					&OPRIMULStyle,
					&OPRRegister3);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
		ModifierGroups[3].Initialize(true, 1, &MRIMULSAT);
	}
}IRIMAD;


struct InstructionRuleISETP: InstructionRule
{
	InstructionRuleISETP(): InstructionRule("ISETP", 2, true, false)
	{
		hpBinaryStringToOpcode8("1100010000111011100000000000000000000000000000000111000000011000", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRRegister1,
					&OPRIADDStyle,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 6, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE);
		ModifierGroups[1].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRISETP;


#else
#define RulesInstructionIntegerDefined
#endif