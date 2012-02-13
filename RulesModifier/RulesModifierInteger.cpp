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

