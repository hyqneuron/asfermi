/*
This file contains instruction rules
*/
#ifndef RulesInstructionDefined



//	6
//-----Specific Instruction Rules
struct INstructionRuleMOV: InstructionRule
{
	INstructionRuleMOV(): InstructionRule("MOV", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0010011110111000000000000000000000000000000000000000000000010100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,
					&OPRMOVStyle);
	}
}IRMOV;
struct InstructionRuleLD: InstructionRule
{
	InstructionRuleLD() : InstructionRule("LD", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,					
					&OPRGlobalMemoryWithImmediate32);
		ModifierGroups[0].Initialize(true, 2,
					&MR64,
					&MR128 );
	}
}IRLD;

struct InstructionRuleST: InstructionRule
{
	InstructionRuleST() : InstructionRule("ST", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRGlobalMemoryWithImmediate32,
					&OPRRegister0);
		ModifierGroups[0].Initialize(true, 2,
					&MR64,
					&MR128 );
	}
}IRST;
struct InstructionRuleEXIT: InstructionRule
{
	InstructionRuleEXIT() : InstructionRule("EXIT", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110011110111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IREXIT;
struct InstructionRuleFADD: InstructionRule
{
	InstructionRuleFADD() : InstructionRule("FADD", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0000000000111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRRegister1,
					&OPRFADDStyle);
	}
}IRFADD;
struct InstructionRuleIADD: InstructionRule
{
	InstructionRuleIADD() : InstructionRule("IADD", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegister0,
					&OPRRegister1,
					&OPRIADDStyle);
	}
}IRIADD;
struct InstructionRuleNOP: InstructionRule
{
	InstructionRuleNOP(): InstructionRule("NOP", 0, true, false)
	{
		SetOperands(1, &OPRIgnored);
		InstructionRule::BinaryStringToOpcode8("0010011110111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
	}
}IRNOP;
struct InstructionRuleISETP: InstructionRule
{
	InstructionRuleISETP(): InstructionRule("ISETP", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100010000111011100000000000000000000000000000000111000000011000", OpcodeWord0, OpcodeWord1);
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

struct InstructionRuleS2R: InstructionRule
{
	InstructionRuleS2R(): InstructionRule("S2R", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0010000000111000000000000000000000000000000000000000000000110100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRRegister0, &OPRS2R);
	}
}IRS2R;

struct InstructionRuleIMUL: InstructionRule
{
	InstructionRuleIMUL(): InstructionRule("IMUL", 3, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRRegister1,
					&OPRIMULStyle);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
	}
}IRIMUL;

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
//-----End of specific instruction rules

#else
#define RulesInstructionDefined
#endif