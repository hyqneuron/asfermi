#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "stdafx.h"

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

