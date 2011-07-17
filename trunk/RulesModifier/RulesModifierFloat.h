#ifndef RulesModifierFloatDefined


struct ModifierRuleFADD32IFTZ: ModifierRule
{
	ModifierRuleFADD32IFTZ(): ModifierRule("FTZ", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111111111111111111111", Mask0);
		Bits0 = 1<<5;
	}
}MRFADD32IFTZ;

struct ModifierRuleFMULR: ModifierRule
{
	ModifierRuleFMULR(int type, char* name): ModifierRule("", false, true, false)
	{
		Name = name;
		::InstructionRule::BinaryStringToOpcode4("11111111111111111111111001111111", Mask1);
		Bits1 = type<<23;
	}
}MRFMULRM(1, "RM"), MRFMULRP(2, "RP"), MRFMULRZ(3, "RZ");



struct ModifierRuleFMULSAT: ModifierRule
{
	ModifierRuleFMULSAT(): ModifierRule("SAT", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111011111111111111111111111111", Mask0);
		Bits0 = 1<<5;
	}
}MRFMULSAT;

#else
#define RulesModifierFloatDefined
#endif