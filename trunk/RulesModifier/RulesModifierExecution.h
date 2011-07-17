#ifndef RulesModifierExecutionDefined


struct ModifierRuleCALNOINC: ModifierRule
{
	ModifierRuleCALNOINC(): ModifierRule("NOINC", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111110111111111111111", Mask0);
		Bits0 = 0;
	}
}MRCALNOINC;

struct ModifierRuleBRAU: ModifierRule
{
	ModifierRuleBRAU(): ModifierRule("U", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111101111111111111111", Mask0);
		Bits0 = 1<<15;
	}
}MRBRAU;
#else
#define RulesModifierExecutionDefined
#endif