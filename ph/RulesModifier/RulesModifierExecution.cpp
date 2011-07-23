#include "RulesModifierExecution.h"
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"


struct ModifierRuleCALNOINC: ModifierRule
{
	ModifierRuleCALNOINC(): ModifierRule("NOINC", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111111111110111111111111111", Mask0);
		Bits0 = 0;
	}
}MRCALNOINC;

struct ModifierRuleBRAU: ModifierRule
{
	ModifierRuleBRAU(): ModifierRule("U", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111111111101111111111111111", Mask0);
		Bits0 = 1<<15;
	}
}MRBRAU;