#include "..\DataTypes.h"
#include "..\GlobalVariables.h"

#include "stdafx.h"

#include "..\RulesOperand.h"
#include "RulesOperandRegister.h"


struct OperandRuleRegister: OperandRule
{
	int Offset;//offset of register bits in OpcodeWord0
	//14 for reg0, 20 for reg1, 26 for reg2. reg3 is not being dealt with here
	
	//this constructor is not really so useful. However, Optional operand can be indicated
	//here with a type Optional instead of Register
	OperandRuleRegister(int offset): OperandRule(Register)
	{
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		//parse the expression using the parsing function defined under SubString
		int result = component.ToRegister();
		//Check if this register is the highest register used so far
		//issue: .128 and .64 will cause the highest register used be higher than the register indicated in the expression
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		//apply result to OpcodeWord0
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord0 |= result;
	}
};
OperandRuleRegister OPRRegister0(14), OPRRegister1(20), OPRRegister2(26);

//reg3 used a separate rule because it applies it result to OpcodeWord1 instead of 0
struct OperandRuleRegister3: OperandRule
{
	OperandRuleRegister3():OperandRule(Register){}
	virtual void Process(SubString &component)
	{
		//parse
		int result = component.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		//apply result
		result = result<<17;
		csCurrentInstruction.OpcodeWord1 |=result;
	}
}OPRRegister3;

//Note that some operands can have modifiers
//This rule deals with registers that can have the .CC modifier
struct OperandRuleRegisterWithCC: OperandRule
{
	int Offset, FlagPos;
	OperandRuleRegisterWithCC(int offset, int flagPos): OperandRule(Register)
	{
		Offset = offset;
		FlagPos = flagPos; //FlagPos is the position of the bit in OpcodeWord1 to be set to 1
	}
	virtual void Process(SubString &component)
	{
		//parse the register expression
		int result = component.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		//apply result
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord0 |= result;
		//look for .CC
		int dotPos = component.Find('.', 0);
		if(dotPos!=-1)
		{
			SubString mod = component.SubStr(dotPos, component.Length - dotPos);
			if(mod.Length>=3 && mod[1] == 'C' && mod[2] == 'C')
			{
				csCurrentInstruction.OpcodeWord1 |= 1<<FlagPos;
			}
		}
	}
}OPRRegisterWithCC4IADD32I(14, 26), OPRRegisterWithCCAt16(14, 16);//for reg0


//Predicate register operand
struct OperandRulePredicate: OperandRule
{
	int Offset; //offset of the predicate's bitfield
	bool Word0; //whether it applies to OpcodeWord0 or Word1
	OperandRulePredicate(int offset, bool word0, bool optional): OperandRule(Predicate)
	{
		//some predicate operands can be optional
		if(optional)
			Type = OperandType::Optional;
		Word0 = word0;
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		//No parsing function in SubString is available to process predicate expression
		//So the parsing is done here
		if(component.Length<2 || (component[0] != 'p' && component[0] != 'P'))
			throw 126; //incorrect predicate
		//pt
		if(component[1]=='t' || component[1] == 'T')
			result = 7;
		//Px
		else
		{
			result = component[1] - 48;
			if(result<0 || result > 7)
				throw 126;
		}
		result <<= Offset;
		//clear the bit field
		csCurrentInstruction.OpcodeWord0 &= ~(7<<Offset);
		//apply result
		if(Word0)
			csCurrentInstruction.OpcodeWord0 |= result;
		else
			csCurrentInstruction.OpcodeWord1 |= result;
	}
}OPRPredicate1(14, true, false), OPRPredicate0(17, true, false), OPRPredicate2NotNegatable(17, false, true);

//Some predicate registers expressions can be negated with !
//this kind of operand is processed separately
struct OperandRulePredicate2: OperandRule
{
	OperandRulePredicate2(): OperandRule(Optional){}
	virtual void Process(SubString &component)
	{
		int startPos = 0;
		if(component[0]=='!')
		{
			startPos = 1;
			csCurrentInstruction.OpcodeWord1 |= 1<<20;
			component.Start++;
			component.Length--;
		}
		OPRPredicate2NotNegatable.Process(component);
		if(startPos)
		{
			component.Start--;
			component.Length++;
		}
	}
}OPRPredicate2;


struct OperandRuleFADD32IReg1: OperandRule
{
	OperandRuleFADD32IReg1(): OperandRule(Register)
	{
	}
	virtual void Process(SubString &component)
	{
		int startPos = 1;
		if(component[0]=='-')
			csCurrentInstruction.OpcodeWord0 |= 1<<9;
		else if(component[0]=='|')
			csCurrentInstruction.OpcodeWord0 |= 1<<7;
		else startPos = 0;
		//leave the operator out when processing it using a general-purpose operand rule
		OPRRegister1.Process(component.SubStr(startPos, component.Length - startPos));
	}
}OPRFADD32IReg1;

