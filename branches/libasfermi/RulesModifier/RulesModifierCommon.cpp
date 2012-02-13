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

#include "RulesModifierCommon.h"


struct ModifierRuleSETPLogic: ModifierRule
{
	ModifierRuleSETPLogic(int type) : ModifierRule("", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111100111111111", Mask1);
		//AND
		if(type==0)
		{
		Name = "AND";
		hpBinaryStringToOpcode4("00000000000000000000000000000000", Bits1);
		}
		//OR
		else if(type==1)
		{
		Name = "OR";
		hpBinaryStringToOpcode4("00000000000000000000010000000000", Bits1);
		}
		//XOR
		else
		{
		Name = "XOR";
		hpBinaryStringToOpcode4("00000000000000000000001000000000", Bits1);
		}
	}
}MRSETPLogicAND(0), MRSETPLogicOR(1), MRSETPLogicXOR(2);

struct ModifierRuleSETPComparison: ModifierRule
{
	ModifierRuleSETPComparison(int type): ModifierRule("", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111000011111", Mask1);
		
		Bits1 = type<<23;
		switch(type)
		{
		case 1: //LT
			Name = "LT";
			break;
		case 2: //EQ
			Name = "EQ";
			break;
		case 3: //LE
			Name = "LE";
			break;
		case 4: //GT
			Name = "GT";
			break;
		case 5: //NE
			Name = "NE";
			break;
		case 6: //GE
			Name = "GE";
			break;
		case 7: //NUM
			Name = "NUM";
			break;
		case 8: //NAN
			Name = "NAN";
			break;
		case 9: //LTU
			Name = "LTU";
			break;
		case 10://EQU
			Name = "EQU";
			break;
		case 11://LEU
			Name = "LEU";
			break;
		case 12://GTU
			Name = "GTU";
			break;
		case 13://NEU
			Name = "NEU";
			break;
		case 14://GEU
			Name = "GEU";
			break;
		default:
			throw exception();
		};
	}
}	MRSETPComparisonLT(1),
	MRSETPComparisonEQ(2),
	MRSETPComparisonLE(3),
	MRSETPComparisonGT(4),
	MRSETPComparisonNE(5),
	MRSETPComparisonGE(6),
	MRSETPComparisonNUM(7),
	MRSETPComparisonNAN(8),
	MRSETPComparisonLTU(9),
	MRSETPComparisonEQU(10),
	MRSETPComparisonLEU(11),
	MRSETPComparisonGTU(12),
	MRSETPComparisonNEU(13),
	MRSETPComparisonGEU(14);

struct ModifierRuleS: ModifierRule
{
	ModifierRuleS(): ModifierRule("S", true, false, false)
	{
		Mask0 = 0xffffffff;
		Bits0 = 1 << 4;
	}
}MRS;

struct ModifierRuleALU: ModifierRule
{
	ModifierRuleALU(int type): ModifierRule("", true, false, false)
	{
		Mask0 = 0xfffffcff;
		Bits0 = type << 8;
		if(type==1)
			Name = "XLU";
		else if(type==2)
			Name = "ALU";
		else throw exception();
	}
}MRALU(2), MRXLU(1);
