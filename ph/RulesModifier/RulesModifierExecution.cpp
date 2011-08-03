
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"

#include "stdafx.h"

#include "RulesModifierExecution.h"


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

struct ModifierRuleNOPTRIG: ModifierRule
{
	ModifierRuleNOPTRIG(): ModifierRule("TRIG", false, true, false)
	{
		Mask1 = 0xffffffff;
		Bits1 = 1 << 18;
	}
}MRNOPTRIG;

struct ModifierRuleNOPOP: ModifierRule
{
	ModifierRuleNOPOP(int type): ModifierRule("", false, true, false)
	{
		hpBinaryStringToOpcode4("11 11111111 11111111 10000111 111111", Mask1);
		Bits1 = type<<19;
		if(type==1)
			Name = "FMA64";
		else if(type == 2)
			Name = "FMA32";
		else if(type == 3)
			Name = "XLU";
		else if(type == 4)
			Name = "ALU";
		else if(type == 5)
			Name = "AGU";
		else if(type == 6)
			Name = "SU";
		else if(type == 7)
			Name = "FU";
		else if(type == 8)
			Name = "FMUL";
	}
}	MRNOPFMA64(1),
	MRNOPFMA32(2),
	MRNOPXLU  (3),
	MRNOPALU  (4),
	MRNOPAGU  (5),
	MRNOPSU   (6),
	MRNOPFU   (7),
	MRNOPFMUL (8);