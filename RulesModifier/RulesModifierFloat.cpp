/*
 * Copyright (c) 2011, 2012 by Hou Yunqing
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

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


struct ModifierRuleFMUL32IFTZ: ModifierRule
{
	ModifierRuleFMUL32IFTZ(): ModifierRule("FTZ", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask0);
		Bits0 = 1<<6;
	}
}MRFMUL32IFTZ;

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


struct ModifierRuleFADDSAT : ModifierRule
{
	ModifierRuleFADDSAT() : ModifierRule("SAT", false, true, false)
	{
		Mask1 = 0xffffffff;
		Bits1 = 1<<17;
	}
}MRFADDSAT;


struct ModifierRuleFMULSAT: ModifierRule
{
	ModifierRuleFMULSAT(): ModifierRule("SAT", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		Bits0 = 1<<5;
	}
}MRFMULSAT;


struct ModifierRuleMUFU: ModifierRule
{
	ModifierRuleMUFU(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("1111 111111 1111 111111 111111 0000 11", Mask0);
		Bits0 = type<<26;
		if(type==0)
			Name = "COS";
		else if(type==1)
			Name = "SIN";
		else if(type==2)
			Name = "EX2";
		else if(type==3)
			Name = "LG2";
		else if(type==4)
			Name = "RCP";
		else if(type==5)
			Name = "RSQ";
		else if(type==6)
			Name = "RCP64H";
		else if(type==7)
			Name = "RSQ64H";
	}
}	MRMUFUCOS(0),
	MRMUFUSIN(1),
	MRMUFUEX2(2),
	MRMUFULG2(3),
	MRMUFURCP(4),
	MRMUFURSQ(5),
	MRMUFURCP64H(6),
	MRMUFURSQ64H(7);
