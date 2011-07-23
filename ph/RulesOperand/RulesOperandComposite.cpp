#include "..\DataTypes.h"
#include "..\GlobalVariables.h"
#include "..\RulesOperand.h"
#include "..\helper\helperMixed.h"
#include "RulesOperandRegister.h"
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
		csMaxReg = (register2 > csMaxReg)? register2: csMaxReg;\
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

