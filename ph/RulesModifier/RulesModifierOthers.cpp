
#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
//#include "stdafx.h" //SMark

#include "RulesModifierOthers.h"


struct ModiferRuleCCTLOp1: ModifierRule
{
	ModiferRuleCCTLOp1(int type): ModifierRule("", true , false, false)
	{
		hpBinaryStringToOpcode4("1111 111111 1111 111111 111111 00 1111", Mask0);
		Bits0 = type<<26;
		if(type==1)
			Name = "U";
		else if(type == 2)
			Name = "C";
		else if(type == 3)
			Name = "I";
		else
			throw exception();
	}
}	MRCCTLOp1U(1),
	MRCCTLOp1C(2),
	MRCCTLOp1I(3);

struct ModifierRuleCCTLOp2: ModifierRule
{
	ModifierRuleCCTLOp2(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 100011 1111 111111 111111 111111", Mask0);
		Bits0 = type<<5;
		if(type==0)
			Name = "QRY1";
		else if(type==1)
			Name = "PF1";
		else if(type==2)
			Name = "PF1_5";
		else if(type==3)
			Name = "PR2";
		else if(type==4)
			Name = "WB";
		else if(type==5)
			Name = "IV";
		else if(type==6)
			Name = "IVALL";
		else if(type==7)
			Name = "RS";
		else
			throw exception();
	}
}	MRCCTLOp2QRY1(0),
	MRCCTLOp2PF1 (1) ,
	MRCCTLOp2PF1_5(2),
	MRCCTLOp2PR2(3),
	MRCCTLOp2WB(4),
	MRCCTLOp2IV(5),
	MRCCTLOp2IVALL(6),
	MRCCTLOp2RS(7);