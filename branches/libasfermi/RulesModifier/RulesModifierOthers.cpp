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

struct ModifierRulePSETPMainop: ModifierRule
{
	ModifierRulePSETPMainop(int type): ModifierRule("", true, false, false)
	{
		Mask0 = 0x3fffffff;
		Bits0 = type << 30;
		if(type==0)
			Name = "AND";
		else if(type==1)
			Name = "OR";
		else if(type == 2)
			Name= "XOR";
		else throw exception();
	}
}MRPSETPAND(0), MRPSETPOR(1), MRPSETPXOR(2);
