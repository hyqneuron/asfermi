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
	InstructionRuleLD() : InstructionRule("LD", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,					
					&OPRGlobalMemoryWithImmediate32);
		ModifierGroups[0].Initialize(true, 3,
					&MRLDCopCG,
					&MRLDCopCS,
					&MRLDCopCV);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8,
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLD;


struct InstructionRuleLDU: InstructionRule
{
	InstructionRuleLDU() : InstructionRule("LDU", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000010001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,					
					&OPRGlobalMemoryWithImmediate32);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8,
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLDU;


struct InstructionRuleLDL: InstructionRule
{
	InstructionRuleLDL() : InstructionRule("LDL", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000000011", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRRegister0,					
					&OPRGlobalMemoryWithImmediate32);
		ModifierGroups[0].Initialize(true, 3,
					&MRLDCopCG,
					&MRLDCopLU,
					&MRLDCopCV);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8,
					&MRLDS8,
					&MRLDU16,
					&MRLDS16,
					&MRLD64,
					&MRLD128);
	}
}IRLDL;

struct InstructionRuleLDS : InstructionRule
{
	InstructionRuleLDS(): InstructionRule("LDS", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000010000011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRSharedMemoryWithImmediate20);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRLDS;

struct InstructionRuleLDC : InstructionRule
{
	InstructionRuleLDC(): InstructionRule("LDC", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("0110000100111000000000000000000000000000000000000000000000101000", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRConstantMemory);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRLDC;

struct InstructionRuleST: InstructionRule
{
	InstructionRuleST() : InstructionRule("ST", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRGlobalMemoryWithImmediate32,
					&OPRRegister0);
		ModifierGroups[0].Initialize(true, 3,
					&MRSTCopCG,
					&MRSTCopCS,
					&MRSTCopWT);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRST;


struct InstructionRuleSTL: InstructionRule
{
	InstructionRuleSTL() : InstructionRule("STL", 2, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000000010011", OpcodeWord0, OpcodeWord1);
		SetOperands(2, 
					&OPRGlobalMemoryWithImmediate32,
					&OPRRegister0);
		ModifierGroups[0].Initialize(true, 3,
					&MRSTCopCG,
					&MRSTCopCS,
					&MRSTCopWT);
		ModifierGroups[1].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRSTL;

struct InstructionRuleSTS : InstructionRule
{
	InstructionRuleSTS(): InstructionRule("STS", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1010000100111000000000000000000000000000000000000000000010010011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRSharedMemoryWithImmediate20,
					&OPRRegister0);
		ModifierGroups[0].Initialize(true, 6,
					&MRLDU8, 
					&MRLDS8, 
					&MRLDU16, 
					&MRLDS16, 
					&MRLD64, 
					&MRLD128);
	}
}IRSTS;

struct InstructionRuleEXIT: InstructionRule
{
	InstructionRuleEXIT() : InstructionRule("EXIT", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110011110111000000000000000000000000000000000000000000000000001", OpcodeWord0, OpcodeWord1);
	}
}IREXIT;

struct InstructionRuleCAL: InstructionRule
{
	InstructionRuleCAL() : InstructionRule("CAL", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110000000000000100000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate24HexConstant);
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRCAL;

struct InstructionRuleBRA: InstructionRule
{
	InstructionRuleBRA() : InstructionRule("BRA", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110011110111000000000000000000000000000000000000000000000000010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate24HexConstant);
		ModifierGroups[0].Initialize(true, 1, &MRBRAU);
	}
}IRBRA;

struct InstructionRulePRET: InstructionRule
{
	InstructionRulePRET() : InstructionRule("PRET", 1, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110000000000001000000000000000000000000000000000000000000011110", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRImmediate24HexConstant);
		ModifierGroups[0].Initialize(true, 1, &MRCALNOINC);
	}
}IRPRET;



struct InstructionRuleRET: InstructionRule
{
	InstructionRuleRET() : InstructionRule("RET", 0, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1110011110111000000000000000000000000000000000000000000000001001", OpcodeWord0, OpcodeWord1);
	}
}IRRET;

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
	InstructionRuleIMUL(): InstructionRule("IMUL", 4, true, false)
	{
		InstructionRule::BinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
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
		InstructionRule::BinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000000100", OpcodeWord0, OpcodeWord1);
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
//-----End of specific instruction rules

#else
#define RulesInstructionDefined
#endif