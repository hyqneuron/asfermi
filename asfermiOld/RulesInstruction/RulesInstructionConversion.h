#ifndef RulesInstructionConversionDefined




struct InstructionRuleF2I: InstructionRule
{
	InstructionRuleF2I(): InstructionRule("F2I", 4, true, false)
	{
		hpBinaryStringToOpcode8("0010000100111000000001001000000000000000000000000000000000101000", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRF2I);
		ModifierGroups[0].Initialize(true, 1, &MRF2IFTZ);
		ModifierGroups[1].Initialize(true, 6,
					&MRF2IDestU16,
					&MRF2IDestU32,
					&MRF2IDestU64,
					&MRF2IDestS16,
					&MRF2IDestS32,
					&MRF2IDestS64);
		ModifierGroups[2].Initialize(true, 3,
					&MRF2ISourceF16,
					&MRF2ISourceF32,
					&MRF2ISourceF64);
		ModifierGroups[3].Initialize(true, 3,
					&MRF2IFLOOR,
					&MRF2ICEIL,
					&MRF2ITRUNC);
	}
}IRF2I;


struct InstructionRuleF2F: InstructionRule
{
	InstructionRuleF2F(): InstructionRule("F2F", 5, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000000 010010 0000000000000000000000 0000000000 001000", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRF2I);
		ModifierGroups[0].Initialize(true, 1, &MRF2IFTZ);
		ModifierGroups[1].Initialize(true, 5,
					&MRI2FDestF16,
					&MRI2FDestF32,
					&MRI2FDestF64,
					&MRF2FPASS,
					&MRF2FROUND);
		ModifierGroups[2].Initialize(true, 3,
					&MRF2ISourceF16,
					&MRF2ISourceF32,
					&MRF2ISourceF64);
		ModifierGroups[3].Initialize(true, 3,
					&MRI2FRM,
					&MRI2FRP,
					&MRI2FRZ);
		ModifierGroups[4].Initialize(true, 1, &MRFMULSAT);
	}
}IRF2F;



struct InstructionRuleI2F: InstructionRule
{
	InstructionRuleI2F(): InstructionRule("I2F", 3, true, false)
	{
		hpBinaryStringToOpcode8("0010000001111000000001001000000000000000000000000000000000011000", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRI2F);
		ModifierGroups[0].Initialize(true, 3,
					&MRI2FDestF16,
					&MRI2FDestF32,
					&MRI2FDestF64);
		ModifierGroups[1].Initialize(true, 6,
					&MRI2FSourceU16,
					&MRI2FSourceU32,
					&MRI2FSourceU64,
					&MRI2FSourceS16,
					&MRI2FSourceS32,
					&MRI2FSourceS64);
		ModifierGroups[2].Initialize(true, 3,
					&MRI2FRM,
					&MRI2FRP,
					&MRI2FRZ);
	}
}IRI2F;

struct InstructionRuleI2I: InstructionRule
{
	InstructionRuleI2I(): InstructionRule("I2I", 3, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000000 010010 0000000000000000000000 0000000000 111000", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRI2F);
		ModifierGroups[0].Initialize(true, 8,
					&MRF2IDestU8,
					&MRF2IDestU16,
					&MRF2IDestU32,
					&MRF2IDestU64,
					&MRF2IDestS8,
					&MRF2IDestS16,
					&MRF2IDestS32,
					&MRF2IDestS64);
		ModifierGroups[1].Initialize(true, 8,
					&MRI2FSourceU8,
					&MRI2FSourceU16,
					&MRI2FSourceU32,
					&MRI2FSourceU64,
					&MRI2FSourceS8,
					&MRI2FSourceS16,
					&MRI2FSourceS32, 
					&MRI2FSourceS64);
		ModifierGroups[2].Initialize(true, 1, &MRFMULSAT);
	}
}IRI2I;

#else
#define RulesInstructionConversionDefined
#endif