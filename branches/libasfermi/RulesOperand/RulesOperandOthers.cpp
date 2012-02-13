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
#include "../GlobalVariables.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "../RulesOperand.h"
#include "RulesOperandComposite.h"
#include "RulesOperandRegister.h"
#include "RulesOperandOthers.h"

//ignored operand: currently used for NOP
struct OperandRuleIgnored: OperandRule
{
	OperandRuleIgnored() : OperandRule(Optional){}
	virtual void Process(SubString &component)
	{
		//do nothing
	}
}OPRIgnored;




struct OperandRule32I: OperandRule
{
	//this constructor is not really so useful. However, Optional operand can be indicated
	//here with a type Optional instead of Custom
	OperandRule32I() : OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		int startPos = 0;
		//floating point number expression
		if(component[0]=='F')
		{
			result = component.ToImmediate32FromFloatConstant();
			goto write;
		}

		//'-' here is not operator. It's part of a constant expression
		if(component[0]=='-')
			startPos=1;
		//hex constant
		if(component.Length-startPos>2 && component[startPos] == '0' && (component[startPos+1]=='x' || component[startPos+1]=='X'))
		{
			result = component.ToImmediate32FromHexConstant(true);
		}
		//int
		else
		{
			result = component.ToImmediate32FromIntConstant();
		}
		write:
		WriteToImmediate32(result);
	}
}OPR32I;





struct OperandRuleLOP: OperandRule
{
	int ModShift;
	OperandRuleLOP(int modShift): OperandRule(Custom)
	{
		ModShift = modShift;
	}
	virtual void Process(SubString &component)
	{
		bool negate = false;
		if(component[0]=='~')
		{
			negate = true;
			component.Start++;
			component.Length--;
		}
		if(component.Length<1)
			throw 132; //empty operand
		if(ModShift==8)
		{
			if(component[0]=='c')
			{
				SetConstMem(component, 0x1f, true);
				MarkConstantMemoryForImmediate32();
			}
			else
				((OperandRule*)&OPRMOVStyle)->Process(component);
		}
		else
			((OperandRule*)&OPRRegister1)->Process(component);
		if(negate)
		{
			csCurrentInstruction.OpcodeWord0 |= 1<<ModShift;
			component.Start--;
			component.Length++;
		}
	}
}OPRLOP1(9), OPRLOP2(8);


struct OperandRuleF2I: OperandRule
{
	bool F2I;
	OperandRuleF2I(bool f2I): OperandRule(Custom)
	{
		F2I = f2I;
	}
	virtual void Process(SubString &component)
	{
		bool operated = false;
		if(component[0]=='-')
		{
			operated = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<8;
			component.Start++;
			component.Length--;
		}
		else if(component[0]=='|')
		{
			operated = true;
			csCurrentInstruction.OpcodeWord0 |= 1<<6;
			component.Start++;
			component.Length--;
		}
		if(F2I)
			((OperandRule*)&OPRFMULStyle)->Process(component);
		else
			((OperandRule*)&OPRIMULStyle)->Process(component);
		if(operated)
		{
			component.Start--;
			component.Length++;
		}
	}
}OPRF2I(true), OPRI2F(false);


struct OperandRuleISCADDShift: OperandRule
{
	OperandRuleISCADDShift(): OperandRule(Custom)
	{
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		if(component.IsHex())
			result = component.ToImmediate32FromHexConstant(false);
		else
			result = component.ToImmediate32FromInt32();
		if(result>=32)
			throw 133;//shift can be no larger than 31
		csCurrentInstruction.OpcodeWord0 |= result << 5; //assumes that the opcode0 has unwritten field
	}
}OPRISCADDShift;


struct OperandRuleNOPCC: OperandRule
{
	bool Initialized;
	SortElement *SortedList;
	unsigned int *IndexList;
	unsigned int ElementCount;
	void Initialize()
	{
		Initialized = true;
		list<SortElement> sElements;

		sElements.push_back(SortElement((void*)0,"F"));
		sElements.push_back(SortElement((void*)1,"LT")); 
		sElements.push_back(SortElement((void*)2,"EQ")); 
		sElements.push_back(SortElement((void*)3,"LE")); 
		sElements.push_back(SortElement((void*)4,"GT")); 
		sElements.push_back(SortElement((void*)5,"NE")); 
		sElements.push_back(SortElement((void*)6,"GE")); 
		sElements.push_back(SortElement((void*)7,"NUM")); 
		sElements.push_back(SortElement((void*)8,"NAN")); 
		sElements.push_back(SortElement((void*)9,"LTU")); 
		sElements.push_back(SortElement((void*)10,"EQU")); 
		sElements.push_back(SortElement((void*)11,"LEU")); 
		sElements.push_back(SortElement((void*)12,"GTU")); 
		sElements.push_back(SortElement((void*)13,"NEU")); 
		sElements.push_back(SortElement((void*)14,"GEU")); 
		sElements.push_back(SortElement((void*)15,"T")); 
		sElements.push_back(SortElement((void*)16,"OFF")); 
		sElements.push_back(SortElement((void*)17,"LO")); 
		sElements.push_back(SortElement((void*)18,"SFF")); 
		sElements.push_back(SortElement((void*)19,"LS")); 
		sElements.push_back(SortElement((void*)20,"HI")); 
		sElements.push_back(SortElement((void*)21,"SFT")); 
		sElements.push_back(SortElement((void*)22,"HS"));
		sElements.push_back(SortElement((void*)23,"OFT")); 
		sElements.push_back(SortElement((void*)24,"CSM_TA")); 
		sElements.push_back(SortElement((void*)25,"CSM_TR")); 
		sElements.push_back(SortElement((void*)26,"CSM_MX")); 
		sElements.push_back(SortElement((void*)27,"FCSM_TA")); 
		sElements.push_back(SortElement((void*)28,"FCSM_TR")); 
		sElements.push_back(SortElement((void*)29,"FCSM_MX")); 
		sElements.push_back(SortElement((void*)30,"RLE")); 
		sElements.push_back(SortElement((void*)31,"RGT")); 
		SortInitialize(sElements, SortedList, IndexList);
		ElementCount = sElements.size();
		Initialized = true;
	}
	OperandRuleNOPCC(): OperandRule(Optional)
	{
		Initialized = false;
	}
	virtual void Process(SubString &component)
	{
		if(component.Length<4 || component[0]!='C' || component[1] != 'C' || component[2] != '.')
			throw 135;//incorrect NOP operand
		SubString mod = component.SubStr(3, component.Length - 3);
		
		if(!Initialized)
			Initialize();

		SortElement found = SortFind(SortedList, IndexList, ElementCount, mod);
		if(found.ExtraInfo==SortNotFound.ExtraInfo)
			throw 135;


		unsigned int type = *((unsigned int*)&found.ExtraInfo);
		csCurrentInstruction.OpcodeWord0 &= ~(15<<5);
		csCurrentInstruction.OpcodeWord0 |= type<<5;
	}
	~OperandRuleNOPCC()
	{
		if(Initialized)
		{
			delete[] IndexList;
			delete[] SortedList;
		}
	}
}OPRNOPCC;
