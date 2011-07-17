#ifndef RulesInstructionFloatDefined



struct InstructionRuleFADD: InstructionRule
{
	InstructionRuleFADD() : InstructionRule("FADD", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0000000000111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRRegister1, //issue: should allow modulus and negate
					&OPRFADDStyle);
	}
}IRFADD;

struct InstructionRuleFADD32I: InstructionRule
{
	InstructionRuleFADD32I() : InstructionRule("FADD32I", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0100000000111000000000000000000000000000000000000000000000010100", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRFADD32IReg1,
					&OPRFADDStyle);
		ModifierGroups[0].Initialize(true, 1, &MRFADD32IFTZ);
	}
}IRFADD32I;



struct InstructionRuleFMUL: InstructionRule
{
	InstructionRuleFMUL(): InstructionRule("FMUL", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0000000000111000000000000000000000000000000000000000000000011010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRRegister1,
					&OPRFMULStyle);
		ModifierGroups[0].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[1].Initialize(true, 1, &MRFMULSAT);
	}
}IRFMUL;


struct InstructionRuleFFMA: InstructionRule
{
	InstructionRuleFFMA(): InstructionRule("FFMA", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0000000000111000000000000000000000000000000000000000000000001100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister0,
					&OPRRegister1,
					&OPRFMULStyle, 
					&OPRRegister3);
		ModifierGroups[0].Initialize(true, 3, &MRFMULRP, &MRFMULRM, &MRFMULRZ);
		ModifierGroups[1].Initialize(true, 1, &MRFMULSAT);
	}
}IRFFMA;

struct InstructionRuleFSETP: InstructionRule
{
	InstructionRuleFSETP(): InstructionRule("FSETP", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100010000111011100000000000000000000000000000000111000000011000", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRRegister1,
					&OPRFADDStyle,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 14, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE,
					&MRSETPComparisonNUM,
					&MRSETPComparisonNAN,
					&MRSETPComparisonLTU,
					&MRSETPComparisonEQU,
					&MRSETPComparisonLEU,
					&MRSETPComparisonGTU,
					&MRSETPComparisonNEU,
					&MRSETPComparisonGEU);
		ModifierGroups[1].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRFSETP;

#else
#define RulesInstructionFloatDefined
#endif