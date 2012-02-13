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

#include "RulesModifierDataMovement.h"


struct ModifierRuleLDType: ModifierRule
{
	ModifierRuleLDType(int type): ModifierRule("", true, false, false)
	{		
		hpBinaryStringToOpcode4("11111000111111111111111111111111", Mask0);
		Bits0 = type<<5;
		switch(type)
		{
		case 0:
			Name = "U8";
			break;
		case 1:
			Name = "S8";
			break;
		case 2:
			Name = "U16";
			break;
		case 3:
			Name = "S16";
			break;
		case 5:
			Name = "64";
			break;
		case 6:
			Name = "128";
			break;
		default:
			throw exception();
			break;
		}
	}
}MRLDU8(0), MRLDS8(1), MRLDU16(2), MRLDS16(3), MRLD64(5), MRLD128(6);

struct ModifierRuleLDCop: ModifierRule
{
	ModifierRuleLDCop(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111001111111111111111111111", Mask0);
		Bits0 = type << 8;
		if(type==1)
			Name = "CG";
		else if(type==2)
			Name = "CS";
		else if(type == 3)
			Name = "CV";
		else if(type == 5)
		{
			Name = "LU";
			Bits0 = 2<<8;
		}
	}
}MRLDCopCG(1),MRLDCopCS(2),MRLDCopCV(3), MRLDCopLU(5);


struct ModifierRuleSTCop: ModifierRule
{
	ModifierRuleSTCop(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111111001111111111111111111111", Mask0);
		Bits0 = type << 8;
		if(type==1)
			Name = "CG";
		else if(type==2)
			Name = "CS";
		else if(type == 3)
			Name = "WT";
	}
}MRSTCopCG(1),MRSTCopCS(2),MRSTCopWT(3);


struct ModifierRuleE: ModifierRule
{
	ModifierRuleE(): ModifierRule("E", false, true, false)
	{
		hpBinaryStringToOpcode4("11111111111111111111111111111111", Mask1);
		Bits1 = 1 << 26;
	}
}MRE;
