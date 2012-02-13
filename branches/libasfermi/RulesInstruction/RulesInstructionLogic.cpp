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


#include "RulesInstructionLogic.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"


struct InstructionRuleLOP: InstructionRule
{
	//InstructionRule(char* name, int modifierGroupCount, bool is8, bool needCustomProcessing)
	InstructionRuleLOP(): InstructionRule("LOP", 1, true, false)
	{
		//set template opcode
		hpBinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010110", OpcodeWord0, OpcodeWord1);
		//Set operands
		//SetOperands(int operandCount, OperandRule*...)
		SetOperands(3,
					&OPRRegister0,
					&OPRLOP1,
					&OPRLOP2);
		//The array ModifierGroups is initialized in the constructor. Now each ModifierGroup needs to be initialized once
		//Initialize(int modifierCount, ModifierRule*...);
		ModifierGroups[0].Initialize(false, 4,
					&MRLOPAND,
					&MRLOPOR,
					&MRLOPXOR,
					&MRLOPPASS);
	}
}IRLOP;


struct InstructionRuleLOP32I: InstructionRule
{
	InstructionRuleLOP32I(): InstructionRule("LOP32I", 1, true, false)
	{
		hpBinaryStringToOpcode8("0100 000000 1110 000000 000000 00000000000000000000000000000000 011100", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegister0,
					&OPRLOP1,
					&OPR32I);
		ModifierGroups[0].Initialize(false, 4,
					&MRLOPAND,
					&MRLOPOR,
					&MRLOPXOR,
					&MRLOPPASS);
	}
}IRLOP32I;


struct InstructionRuleSHR: InstructionRule
{
	InstructionRuleSHR(bool shr) : InstructionRule("", 3, true, false)
	{
		if(shr)
		{
			hpBinaryStringToOpcode8("1100 010001 1110 000000 000000 0000000000000000000000 0000000000 011010", OpcodeWord0, OpcodeWord1);
			Name = "SHR";
		}
		else
		{
			hpBinaryStringToOpcode8("1100 010001 1110 000000 000000 0000000000000000000000 0000000000 000110", OpcodeWord0, OpcodeWord1);
			Name = "SHL";
		}
		SetOperands(3,
					&OPRRegisterWithCCAt16, 
					&OPRRegister1,
					&OPRIADDStyle);
		ModifierGroups[0].Initialize(true, 1, &MRSHRU32);
		ModifierGroups[1].Initialize(true, 1, &MRSHRW);
		ModifierGroups[2].Initialize(true, 1, &MRS);
	}
}IRSHR(true), IRSHL(false);


struct InstructionRuleBFE: InstructionRule
{
	InstructionRuleBFE(): InstructionRule("BFE", 2, true ,false)
	{
		hpBinaryStringToOpcode8("1100 010000 1110 000000 000000 0000000000000000000000 0000000000 001110", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegisterWithCCAt16,
					&OPRRegister1,
					&OPRIMULStyle);
		ModifierGroups[0].Initialize(true, 1, &MRSHRU32);
		ModifierGroups[1].Initialize(true, 1, &MRBFEBREV);
	}
}IRBFE;

struct InstructionRuleBFI: InstructionRule
{
	InstructionRuleBFI(): InstructionRule("BFI", 0, true, false)
	{
		hpBinaryStringToOpcode8("1100 000000 1110 000000 000000 0000000000000000000000 0 000000 000 010100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegisterWithCCAt16,
					&OPRRegister1,
					&OPRIMULStyle,
					&OPRRegister3ForCMP);
	}
}IRBFI;


struct InstructionRuleSEL: InstructionRule
{
	InstructionRuleSEL(): InstructionRule("SEL", 0, true, false)
	{
		hpBinaryStringToOpcode8("0010 000000 1110 000000 000000 0000000000000000000000 0 0000 00000 000100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,					
					&OPRRegister0,
					&OPRRegister1,
					&OPRIMULStyle,
					&OPRPredicate2);
	}
}IRSEL;
