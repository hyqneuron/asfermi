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

#include "RulesInstructionMiscellaneous.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"




struct InstructionRuleS2R: InstructionRule
{
	InstructionRuleS2R(): InstructionRule("S2R", 0, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 111000000000000000000000000000000000000000000000110100", OpcodeWord0, OpcodeWord1);
		SetOperands(2, &OPRRegister0, &OPRS2R);
	}
}IRS2R;

struct InstructionRuleLEPC: InstructionRule
{
	InstructionRuleLEPC(): InstructionRule("LEPC",0, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 111000000000000000000000000000000000000000000000 100010", OpcodeWord0, OpcodeWord1);
		SetOperands(1, &OPRRegister0);
	}

}IRLEPC;

struct InstructionRuleCCTL: InstructionRule
{
	InstructionRuleCCTL(): InstructionRule("CCTL", 3, true, false)
	{
		hpBinaryStringToOpcode8("1010 000000 1110 000000 000000 00 000000000000000000000000000000 0 11001", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRGlobalMemoryWithLastWithoutLast2Bits);
		ModifierGroups[0].Initialize(true, 1, &MRE);
		ModifierGroups[1].Initialize(true, 3,
									&MRCCTLOp1U,
									&MRCCTLOp1C,
									&MRCCTLOp1I);
		ModifierGroups[2].Initialize(false, 8,
									&MRCCTLOp2QRY1,
									&MRCCTLOp2PF1,
									&MRCCTLOp2PF1_5,
									&MRCCTLOp2PR2,
									&MRCCTLOp2WB,
									&MRCCTLOp2IV,
									&MRCCTLOp2IVALL,
									&MRCCTLOp2RS);
	}
}IRCCTL;


struct InstructionRuleCCTLL: InstructionRule
{
	InstructionRuleCCTLL(): InstructionRule("CCTLL", 1, true, false)
	{
		hpBinaryStringToOpcode8("1010 000000 1110 000000 000000 000000000000000000000000 00000000 001011", OpcodeWord0, OpcodeWord1);
		SetOperands(2,
					&OPRRegister0,
					&OPRGlobalMemoryWithImmediate24);
		ModifierGroups[0].Initialize(false, 8,
									&MRCCTLOp2QRY1,
									&MRCCTLOp2PF1,
									&MRCCTLOp2PF1_5,
									&MRCCTLOp2PR2,
									&MRCCTLOp2WB,
									&MRCCTLOp2IV,
									&MRCCTLOp2IVALL,
									&MRCCTLOp2RS);
	}
}IRCCTLL;

struct InstructionRulePSETP: InstructionRule
{
	InstructionRulePSETP(): InstructionRule("PSETP", 2, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000 000 1110 00 0000 00 00000000000000000 0000 00000 110000", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRPredicate2,
					&OPRPredicate3ForPSETP,
					&OPRPredicate1ForVOTE);
		
		ModifierGroups[0].Initialize(true, 3,
					&MRPSETPAND,
					&MRPSETPOR,
					&MRPSETPXOR);
		ModifierGroups[1].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRPSETP;
