#include "RulesModifierConversion.h"
#include "..\DataTypes.h"
#include "..\helper\helperMixed.h"



struct ModifierRuleF2IDest: ModifierRule
{
	ModifierRuleF2IDest(int type, bool sign): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111110111111111111001111111111", Mask0);
		Bits0 = type << 20;
		Bits0 |= (int)sign << 7;
		if(sign)
		{
			if(type==1)
				Name = "S16";
			else if(type==2)
				Name = "S32";
			else if(type==3)
				Name = "S64";
			else
				throw exception("Wrong type");
		}
		else
		{
			if(type==1)
				Name = "U16";
			else if(type==2)
				Name = "U32";
			else if(type==3)
				Name = "U64";
			else
				throw exception("Wrong type");
		}
	}
}MRF2IDestU16(1, false), MRF2IDestU32(2, false), MRF2IDestU64(3, false), MRF2IDestS16(1, true), MRF2IDestS32(2, true), MRF2IDestS64(3, true);

struct ModifierRuleF2ISource: ModifierRule
{
	ModifierRuleF2ISource(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 111111 1111 111111 111001 111111", Mask0);
		Bits0 = type<<23;
		if(type==1)
			Name = "F16";
		else if(type==2)
			Name = "F32";
		else if(type==3)
			Name = "F64";
		else throw exception("Wrong type");
	}
}MRF2ISourceF16(1),MRF2ISourceF32(2),MRF2ISourceF64(3);

struct ModifierRuleF2IRound: ModifierRule
{
	ModifierRuleF2IRound(int type): ModifierRule("", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111001111111111111", Mask1);
		Bits1 = type <<17;
		if(type==1)
			Name = "FLOOR";
		else if(type==2)
			Name = "CEIL";
		else if(type==3)
			Name = "TRUNC";
		else throw exception("Wrong type");
	}
}MRF2IFLOOR(1), MRF2ICEIL(2), MRF2ITRUNC(3);

struct ModifierRuleF2IFTZ: ModifierRule
{
	ModifierRuleF2IFTZ(): ModifierRule("FTZ", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask1);
		Bits1 = 1 <<23;
	}
}MRF2IFTZ;



struct ModifierRuleI2FSource: ModifierRule
{
	ModifierRuleI2FSource(int type, bool sign): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 111110 1111 111111 111001 111111", Mask0);
		Bits0 = type << 23;
		Bits0 |= (int)sign << 9;
		if(sign)
		{
			if(type==0)
				Name = "S8";
			else if(type==1)
				Name = "S16";
			else if(type==2)
				Name = "S32";
			else if(type==3)
				Name = "S64";
			else
				throw exception("Wrong type");
		}
		else
		{
			if(type==0)
				Name = "U8";
			else if(type==1)
				Name = "U16";
			else if(type==2)
				Name = "U32";
			else if(type==3)
				Name = "U64";
			else
				throw exception("Wrong type");
		}
	}
}MRI2FSourceU16(1, false), MRI2FSourceU32(2, false), MRI2FSourceU64(3, false), MRI2FSourceS16(1, true), MRI2FSourceS32(2, true), MRI2FSourceS64(3, true);

struct ModifierRuleI2FDest: ModifierRule
{
	ModifierRuleI2FDest(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111001111111111", Mask0);
		Bits0 = type<<20;
		if(type==1)
			Name = "F16";
		else if(type==2)
			Name = "F32";
		else if(type==3)
			Name = "F64";
		else throw exception("Wrong type");
	}
}MRI2FDestF16(1),MRI2FDestF32(2),MRI2FDestF64(3);

struct ModifierRuleI2FRound: ModifierRule
{
	ModifierRuleI2FRound(int type): ModifierRule("", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111001111111111111", Mask1);
		Bits1 = type <<17;
		if(type==1)
			Name = "RM";
		else if(type==2)
			Name = "RP";
		else if(type==3)
			Name = "RZ";
		else throw exception("Wrong type");
	}
}MRI2FRM(1), MRI2FRP(2), MRI2FRZ(3);
