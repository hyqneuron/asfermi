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

#include "RulesInstructionInteger.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"



struct InstructionRuleIADD: InstructionRule
{
	InstructionRuleIADD() : InstructionRule("IADD", 2, true, false)
	{
		hpBinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegisterWithCCAt16,
					&OPRIMADReg1,
					&OPRIADDStyle);
		ModifierGroups[0].Initialize(true, 1, &MRIADD32ISAT);
		ModifierGroups[1].Initialize(true, 1, &MRIADD32IX);
	}
}IRIADD;


struct INstructionRuleIADD32I: InstructionRule
{
	INstructionRuleIADD32I(): InstructionRule("IADD32I", 2, true, false)
	{
		hpBinaryStringToOpcode8("0100000000111000000000000000000000000000000000000000000000010000", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegisterWithCC4IADD32I,
          				&OPRRegister1,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 1, &MRIADD32ISAT);
		ModifierGroups[1].Initialize(true, 1, &MRIADD32IX);
	}
}IRIADD32I;


struct InstructionRuleIMUL: InstructionRule
{
	InstructionRuleIMUL(): InstructionRule("IMUL", 3, true, false)
	{
		hpBinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegisterWithCCAt16,
					&OPRRegister1,
					&OPRIMULStyle);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
	}
}IRIMUL;

struct InstructionRuleIMUL32I: InstructionRule
{
	InstructionRuleIMUL32I() : InstructionRule("IMUL32I", 3, true, false)
	{
		hpBinaryStringToOpcode8("0100 010100 1110 000000 000000 00000000000000000000000000000000 0 01000", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegisterWithCC4IADD32I, //different cc pos
					&OPRRegister1,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
	}
}IRIMUL32I;

struct InstructionRuleIMAD: InstructionRule
{
	InstructionRuleIMAD(): InstructionRule("IMAD", 4, true, false)
	{
		hpBinaryStringToOpcode8("1100 010100111000000000000000000000000000000000000000000000 000100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegisterWithCCAt16,
					&OPRIMADReg1,
					&OPRIMULStyle,
					&OPRRegister3ForMAD);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
		ModifierGroups[3].Initialize(true, 1, &MRIMULSAT);
	}
}IRIMAD;

struct InstructionRuleISCADD: InstructionRule
{
	InstructionRuleISCADD(): InstructionRule("ISCADD", 0, true, false)
	{
		hpBinaryStringToOpcode8("1100 0 00000 1110 000000 000000 0000000000000000000000 0000000000 000010", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegisterWithCCAt16,
					&OPRISCADDReg1,
					&OPRISCADDAllowNegative,
					&OPRISCADDShift);
	}
}IRISCADD;

struct InstructionRuleISETP: InstructionRule
{
	InstructionRuleISETP(): InstructionRule("ISETP", 3, true, false)
	{
		hpBinaryStringToOpcode8("1100 010000 1110 111 000 000000 0000000000000000000000 0 1110 000000 11000", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRRegister1,
					&OPRIADDStyle,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 6, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE);
		ModifierGroups[1].Initialize(true, 1, &MRISETPU32);
		ModifierGroups[2].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRISETP;

struct InstructionRuleICMP: InstructionRule
{
	InstructionRuleICMP(): InstructionRule("ICMP", 2, true, false)
	{
		hpBinaryStringToOpcode8("1100 010000 1110 000000 000000 0000000000000000000000 0 000000  000 001100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister0,
					&OPRRegister1,
					&OPRIMULStyle,
					&OPRRegister3ForCMP);
					
		ModifierGroups[0].Initialize(false, 6, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE);
		ModifierGroups[1].Initialize(true, 1, &MRIMUL1U32);
	}
}IRICMP;

