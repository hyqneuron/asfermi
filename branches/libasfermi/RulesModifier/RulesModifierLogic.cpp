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

#include "RulesModifierLogic.h"


struct ModifierRuleLOP: ModifierRule
{
	ModifierRuleLOP(int type): ModifierRule("", true, false, false)
	{
		hpBinaryStringToOpcode4("11111100111111111111111111111111", Mask0);
		Bits0 = type << 6;
		if(type==0)
			Name = "AND";
		else if(type==1)
			Name = "OR";
		else if(type==2)
			Name = "XOR";
		else if(type ==3)
			Name = "PASS_B";
	}
}MRLOPAND(0), MRLOPOR(1), MRLOPXOR(2), MRLOPPASS(3);

struct ModifierRuleSHR: ModifierRule
{
	ModifierRuleSHR(bool u32): ModifierRule("", true, false, false)
	{
		Bits0 = 0;
		if(u32)
		{
			Name = "U32";
			hpBinaryStringToOpcode4("1111 101111 1111111111111111111111", Mask0);
		}
		else
		{
			Name = "W";
			hpBinaryStringToOpcode4("1111 111110 1111111111111111111111", Mask0);
		}
	}
}MRSHRU32(true), MRSHRW(false);

struct ModifierRuleBFEBREV :ModifierRule
{
	ModifierRuleBFEBREV(): ModifierRule("BREV", true, false, false)
	{
		Mask0 = 0xffffffff;
		Bits0 = 1 << 8;
	}
}MRBFEBREV;
