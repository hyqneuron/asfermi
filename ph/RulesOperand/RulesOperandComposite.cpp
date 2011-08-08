
#include "../DataTypes.h"
#include "../GlobalVariables.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "../RulesOperand.h"
#include "RulesOperandRegister.h"
#include "RulesOperandConstant.h"
#include "RulesOperandComposite.h"


struct OperandRuleMOVStyle: OperandRule
{
	OperandRuleMOVStyle() : OperandRule(MOVStyle){}

	virtual void Process(SubString &component)
	{
		//Register
		if(component[0] == 'R' || component[0] == 'r')
			((OperandRule*)&OPRRegister2)->Process(component);
		//Constant memory
		else if(component[0]=='c'||component[0]=='C')
		{
			unsigned int bank, memory;
			int register1;
			component.ToConstantMemory(bank, register1, memory);
			if(register1 != 63)
				throw 112; //register cannot be used in MOV-style constant address
			csCurrentInstruction.OpcodeWord1 |= bank<<10;
			WriteToImmediate32(memory);
			MarkConstantMemoryForImmediate32();
		}
		//constant
		else
		{
			unsigned int result;
			//hex
			if(component.Length>2 && component[0]=='0' && (component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate20FromHexConstant(false); //does not allows negative as highest bit cannot be included
			//float
			else if(component[0]=='F')
				result = component.ToImmediate20FromFloatConstant();
			//int
			else
				result = component.ToImmediate20FromIntConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
	}
}OPRMOVStyle;


#define mArithmeticCommonEnd {if(negative){component.Start--;	component.Length++;	}}

#define mArithmeticCommon\
	bool negative = false;\
	if(component[0] == '-')\
	{\
		if(!AllowNegative)\
			throw 129;\
		negative = true;\
		csCurrentInstruction.OpcodeWord0 |= 1<<8; \
		component.Start++;\
		component.Length--;\
		if(component.Length<1) \
			throw 116; \
	}\
	/*register*/\
	if(component[0]=='R' || component[0]=='r')\
	{\
		int register2 = component.ToRegister();\
		if(register2!=63)csMaxReg = (register2 > csMaxReg)? register2: csMaxReg;\
		csCurrentInstruction.OpcodeWord0 |= register2 << 26;\
		MarkRegisterForImmediate32();\
	}		\
	/*constant memory*/\
	else if(component[0]=='c' || component[0] == 'C') \
	{\
		unsigned int bank, memory;\
		int register1;\
		component.ToConstantMemory(bank, register1, memory);\
		if(register1 != 63)\
			throw 112;\
		csCurrentInstruction.OpcodeWord1 |= bank<<10;\
		WriteToImmediate32(memory);\
		MarkConstantMemoryForImmediate32();\
	}
//FADD: Register, Constant memory without reg, 20-bit Float)
//issue: will not work if the second operand, a register, has a negative sign before it. This is, however, possible
//since it's been observed in the output of cuobjdump. The mod1:0 bits, when = 10, negate third operand; when = 01, negate second operand;
//when = 11, produces.PO, which I do not understand for now
struct OperandRuleFADDStyle: OperandRule
{
	bool AllowNegative;
	OperandRuleFADDStyle(bool allowNegative) :OperandRule(FADDStyle)
	{
		AllowNegative = allowNegative;
	}
	virtual void Process(SubString &component)
	{
		//Register or constant memory
		mArithmeticCommon
		//constants
		else
		{
			unsigned int result;
			//float
			if(component[0]=='F')
				result = component.ToImmediate20FromFloatConstant();
			//hex
			else if(component.Length>2 && component[0]=='0' &&(component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate20FromHexConstant(true);
			else
				throw 116;
			
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}
		mArithmeticCommonEnd;
	}
}OPRFADDStyle(true), OPRFMULStyle(false);

struct OperandRuleFAllowNegative: OperandRule
{
	bool OnWord0;
	int FlagPos;
	OperandRuleFAllowNegative(bool onWord0, int flagPos): OperandRule(Custom)
	{
		OnWord0 = onWord0;
		FlagPos = flagPos;
	}
	virtual void Process(SubString &component)
	{
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			int flag = 1<<FlagPos;
			if(OnWord0)
				csCurrentInstruction.OpcodeWord0 |= flag;
			else
				csCurrentInstruction.OpcodeWord1 |= flag;
			if(component.Length<2)
				throw 116;//issue: bad error message
		}
		OPRFMULStyle.Process(component);
		mArithmeticCommonEnd
	}
}OPRFMULAllowNegative(false, 25), OPRFFMAAllowNegative(true, 9);


//IADD: Register, Constant memory without reg, 20-bit Int)
struct OperandRuleIADDStyle: OperandRule
{
	bool AllowNegative;
	OperandRuleIADDStyle(bool allowNegative) :OperandRule(IADDStyle)
	{
		AllowNegative = allowNegative;
	}
	virtual void Process(SubString &component)
	{
		bool allowNegate = true;
		//register or constant memory
		mArithmeticCommon
		//constants
		else
		{
			unsigned int result;
			//hex
			if(component.Length>2 && component[0]=='0' &&(component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate20FromHexConstant(true);
			//int
			else
				result = component.ToImmediate20FromIntConstant();
			WriteToImmediate32(result);
			MarkImmediate20ForImmediate32();
		}	
		mArithmeticCommonEnd
	}
}OPRIADDStyle(true), OPRIMULStyle(false);


struct OperandRuleIAllowNegative: OperandRule
{
	bool OnWord0;
	int FlagPos;
	OperandRuleIAllowNegative(bool onWord0, int flagPos): OperandRule(Custom)
	{
		OnWord0 = onWord0;
		FlagPos = flagPos;
	}
	virtual void Process(SubString &component)
	{
		bool negative = false;
		if(component[0]=='-')
		{
			negative = true;
			component.Start++;
			component.Length--;
			int flag = 1<<FlagPos;
			if(OnWord0)
				csCurrentInstruction.OpcodeWord0 |= flag;
			else
				csCurrentInstruction.OpcodeWord1 |= flag;
			if(component.Length<2)
				throw 116;//issue: bad error message
		}
		OPRIMULStyle.Process(component);
		mArithmeticCommonEnd
	}
}OPRISCADDAllowNegative(false, 23);

struct OperandRuleFADDCompositeWithOperator: OperandRule
{
	OperandRuleFADDCompositeWithOperator(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		int startPos = 1;
		if(component[0]=='-')
			csCurrentInstruction.OpcodeWord0 |= 1<<8;
		else if(component[0]=='|')
			csCurrentInstruction.OpcodeWord0 |= 1<<6;
		else startPos = 0;
		SubString s = component.SubStr(startPos, component.Length - startPos);
		OPRFMULStyle.Process(s);
	}
}OPRFADDCompositeWithOperator;

struct OperandRuleInstructionAddress: OperandRule
{
	OperandRuleInstructionAddress(): OperandRule(Custom)
	{
	}
	virtual void Process(SubString &component)
	{
		//constant memory
		if(component[0]=='c'||component[0]=='C')
		{
			csCurrentInstruction.OpcodeWord0 |= 1 << 14;
			unsigned int bank, memory;
			int reg;
			component.ToConstantMemory(bank, reg, memory);
			if(reg!=63)
				throw 137; //does not accept register
			csCurrentInstruction.OpcodeWord1 |= bank<<10;
			WriteToImmediate32(memory);
		}
		else
		{
			((OperandRule*)&OPRImmediate24HexConstant)->Process(component);
		}
	}
}OPRInstructionAddress;



struct OperandRuleBAR: OperandRule
{
	OperandRuleBAR(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		//register
		if(component[0]=='R')
		{
			result = component.ToRegister();
		}
		//numerical expression
		else
		{
			//hex
			if(component.Length>2&&component[0]=='0'&&(component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate32FromHexConstant(false);
			//int
			else
				result = component.ToImmediate32FromInt32();
			if(result>63)
				throw 139;//too large barrier identifier
			csCurrentInstruction.OpcodeWord1|= 1<<15;
			if(result>csMaxBar)
				csMaxBar = result;
		}
		csCurrentInstruction.OpcodeWord0 |= result << 20;
	}
}OPRBAR;

struct OperandRuleTCount: OperandRule
{
	OperandRuleTCount(): OperandRule(Custom){}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		//register
		if(component[0]=='R')
		{
			result = component.ToRegister();
		}
		//numerical expression
		else
		{
			//hex
			if(component.Length>2&&component[0]=='0'&&(component[1]=='x'||component[1]=='X'))
				result = component.ToImmediate32FromHexConstant(false);
			//int
			else
				result = component.ToImmediate32FromInt32();
			if(result>0xfff)
				throw 140;//thread count should be no larger than 4095
			csCurrentInstruction.OpcodeWord1|= 1<<14;
		}
		csCurrentInstruction.OpcodeWord0 &= 0x03ffffff;
		csCurrentInstruction.OpcodeWord0 |= result << 26;		
		csCurrentInstruction.OpcodeWord0 |= result >> 6;
	}
}OPRTCount;