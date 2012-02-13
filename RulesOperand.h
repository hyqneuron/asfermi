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

#include "DataTypes.h"
#include "GlobalVariables.h"


#ifndef RulesOperandDefined
#define RulesOperandDefined

inline void WriteToImmediate32(unsigned int content)
{
	csCurrentInstruction.OpcodeWord0 |= content <<26;
	csCurrentInstruction.OpcodeWord1 |= content >> 6;
}
inline void MarkConstantMemoryForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 1<<14; //constant memory flag
}
inline void MarkImmediate20ForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 3<<14; //20-bit immediate flag
}
inline void MarkRegisterForImmediate32()
{
}

inline void CheckRegCount(int reg)
{
	if(reg!=63&&reg>=csRegCount)
		csRegCount=reg+1;
}

inline void SetConstMem(SubString &component, int maxBank=0xf, bool specialLast2=false)
{	
	unsigned int bank, memory;
	int register1;
	component.ToConstantMemory(bank, register1, memory, maxBank); //correct
	if(register1 != 63)
		throw 112;  //register cannot be used in composite constant memory operand
	if(specialLast2)
	{
		if(memory%2!=0)
			throw 148; // should be multiples of 4, multiples of 2 allowed for experiment
		if(memory%4!=0)
			::hpWarning(14); //should be multiples of 4
		if(bank>0xf)
		{
			memory |= 1;
			bank -= 0x10;
		}
	}
	csCurrentInstruction.OpcodeWord1 |= bank<<10;
	WriteToImmediate32(memory);
	//MarkConstantMemoryForImmediate32();
}

#include "RulesOperand/RulesOperandConstant.h"
#include "RulesOperand/RulesOperandRegister.h"
#include "RulesOperand/RulesOperandMemory.h"
#include "RulesOperand/RulesOperandComposite.h"
#include "RulesOperand/RulesOperandOthers.h"

#endif
