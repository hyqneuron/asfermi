#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesModifierInteger.h"



struct ModifierRuleIMUL0U32: ModifierRule
{
	ModifierRuleIMUL0U32(): ModifierRule("U32", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101011 1111111111111111111111", Mask0);
		Bits0 = 0;
	}
}MRIMUL0U32;

struct ModifierRuleIMUL1U32: ModifierRule
{
	ModifierRuleIMUL1U32(): ModifierRule("U32", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		Bits0 = 0;
	}
}MRIMUL1U32;


struct ModifierRuleIMUL0S32: ModifierRule
{
	ModifierRuleIMUL0S32(): ModifierRule("S32", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101011 1111111111111111111111", Mask0);
		hpBinaryStringToOpcode4("0000 010100 0000000000000000000000", Bits0);
	}
}MRIMUL0S32;

struct ModifierRuleIMUL1S32: ModifierRule
{
	ModifierRuleIMUL1S32(): ModifierRule("S32", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 1011111 111111111111111111111", Mask0);
		hpBinaryStringToOpcode4("0000 0100000 000000000000000000000", Bits0);
	}
}MRIMUL1S32;

struct ModifierRuleIMULHI: ModifierRule
{
	ModifierRuleIMULHI(): ModifierRule("HI", true, false, false)
	{
		hpBinaryStringToOpcode4("11111101111111111111111111111111", Mask0);
		hpBinaryStringToOpcode4("00000010000000000000000000000000", Bits0);
	}
}MRIMULHI;

struct ModifierRuleIMULSAT: ModifierRule
{
	ModifierRuleIMULSAT(): ModifierRule("SAT", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111101 111111", Mask1);
		hpBinaryStringToOpcode4("00000000000000000000000010 000000", Bits1);
	}
}MRIMULSAT;

struct ModifierRuleIADD32ISAT: ModifierRule
{
	ModifierRuleIADD32ISAT(): ModifierRule("SAT", true, false, false)
	{
		hpBinaryStringToOpcode4("11111011111111111111111111111111", Mask0);
		Bits0 = 1 << 5;
	}
}MRIADD32ISAT;

struct ModifierRuleIADD32IX: ModifierRule
{
	ModifierRuleIADD32IX(): ModifierRule("X", true, false, false)
	{
		hpBinaryStringToOpcode4("11111101111111111111111111111111", Mask0);
		Bits0 = 1 << 6;
	}
}MRIADD32IX;

struct ModifierRuleISETPU32: ModifierRule
{
	ModifierRuleISETPU32(): ModifierRule("U32", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		Bits0 = 0;
	}
}MRISETPU32;

struct ModifierRuleVADD_UD: ModifierRule
{
	ModifierRuleVADD_UD(): ModifierRule("UD", false, true, false)
	{
		hpBinaryStringToOpcode4("1111111111 0111111 111111 111 111111", Mask1);
		Bits1 = 0;
	}
}MRVADD_UD;

struct ModifierRuleVADD_OpType: ModifierRule
{
	//Op1's subwOffset is 12; Op2's subwOffset is 0. Both are in OpcodeWord1
	//Op1's signOffset is 6, 0=Uxx and 1=Sxx, bit resides in OpcodeWord0
	//Op2's signOffset is 5, 0=Uxx and 1=Sxx
	//subw=0 for x8, 4 for x16 and 6 for x32
	ModifierRuleVADD_OpType(char* name, int signOffset, int sign, int subwOffset, int subw): ModifierRule(name, true, true, false)
	{
		Mask0 = 0xFFFFFFFF ^ 1 << signOffset;
		Bits0 = sign << signOffset;
		Mask1 = 0xFFFFFFFF ^ 7 << subwOffset; //subw takes 3 bits
		Bits1 = subw << subwOffset;
	}
};
ModifierRuleVADD_OpType
	MRVADD_Op1_U8("U8", 6, 0, 12, 0),
	MRVADD_Op1_U16("U16", 6, 0, 12, 4),
	MRVADD_Op1_U32("U32", 6, 0, 12, 6),
	MRVADD_Op1_S8("S8", 6, 1, 12, 0),
	MRVADD_Op1_S16("S16", 6, 1, 12, 4),
	MRVADD_Op1_S32("S32", 6, 1, 12, 6);
ModifierRuleVADD_OpType
	MRVADD_Op2_U8("U8", 5, 0, 0, 0),
	MRVADD_Op2_U16("U16", 5, 0, 0, 4),
	MRVADD_Op2_U32("U32", 5, 0, 0, 6),
	MRVADD_Op2_S8("S8", 5, 1, 0, 0),
	MRVADD_Op2_S16("S16", 5, 1, 0, 4),
	MRVADD_Op2_S32("S32", 5, 1, 0, 6);

struct ModifierRuleVADD_SAT: ModifierRule
{
	ModifierRuleVADD_SAT(): ModifierRule("SAT", true, false, false)
	{
		Mask0 = ~(1<<9);
		Bits0 = 1<<9;
	}
}MRVADD_SAT;


struct ModifierRuleVADD_SecOp: ModifierRule
{
	ModifierRuleVADD_SecOp(char* name, int value): ModifierRule(name, false, true, false)
	{
		Mask1 = ~(7<<23);
		Bits1 = value<<23;
	}
};
ModifierRuleVADD_SecOp
	MRVADD_SecOp_MRG_16H("MRG_16H", 0),
	MRVADD_SecOp_MRG_16L("MRG_16L", 1),
	MRVADD_SecOp_MRG_8B0("MRG_8B0", 2),
	MRVADD_SecOp_MRG_8B2("MRG_8B2", 3),
	MRVADD_SecOp_ACC("ACC", 4),
	MRVADD_SecOp_MIN("MIN", 5),
	MRVADD_SecOp_MAX("MAX", 6);

