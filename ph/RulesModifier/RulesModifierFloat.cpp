
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"

#include "stdafx.h"

#include "RulesModifierFloat.h"




struct ModifierRuleFADD32IFTZ: ModifierRule
{
	//ModifierRule(char *name, bool apply0, bool apply1, bool needCustomProcessing)
	ModifierRuleFADD32IFTZ(): ModifierRule("FTZ", true, false, false)
	{
		//Setting the mask. No bits are to be cleared for FTZ, so it's just all 1s
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask0);
		//mod 4 is to be set to 1
		Bits0 = 1<<5;
	}
}MRFADD32IFTZ;

struct ModifierRuleFMULR: ModifierRule
{
	ModifierRuleFMULR(int type, char* name): ModifierRule("", false, true, false)
	{
		Name = name;
		//2 bits are to be cleared
		hpBinaryStringToOpcode4("11111111111111111111111001111111", Mask1);
		//immeb 1:2 to be set to 10, 01 or 11
		Bits1 = type<<23;
	}
}MRFMULRM(1, "RM"), MRFMULRP(2, "RP"), MRFMULRZ(3, "RZ");



struct ModifierRuleFMULSAT: ModifierRule
{
	ModifierRuleFMULSAT(): ModifierRule("SAT", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		Bits0 = 1<<5;
	}
}MRFMULSAT;
