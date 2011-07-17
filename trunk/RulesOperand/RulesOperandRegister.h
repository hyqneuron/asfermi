#ifndef RulesOperandRegisterDefined



struct OperandRuleRegister: OperandRule
{
	int Offset;
	OperandRuleRegister(): OperandRule(Register){}
	OperandRuleRegister(int offset)
	{
		Type = (OperandType)0;
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		int result = component.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		result = result<<Offset;
		csCurrentInstruction.OpcodeWord0 |= result;
	}
};
OperandRuleRegister OPRRegister0(14), OPRRegister1(20), OPRRegister2(26); //different location

struct OperandRuleRegister3: OperandRule
{
	OperandRuleRegister3():OperandRule(Register){}
	virtual void Process(SubString &component)
	{
		int result = component.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
		result = result<<17;
		csCurrentInstruction.OpcodeWord1 |=result;
	}
}OPRRegister3;

struct OperandRuleRegisterWithCC: OperandRule
{
	int Offset, FlagPos;
	OperandRuleRegisterWithCC(int offset, int flagPos): OperandRule(Register)
	{
		Offset = offset;
		FlagPos = flagPos;
	}
	virtual void Process(SubString &component)
	{
		int result = component.ToRegister();
		csMaxReg = (result > csMaxReg)? result: csMaxReg;
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
}OPRRegisterWithCC4IADD32I(14, 26);


//Predicate register operand
struct OperandRulePredicate: OperandRule
{
	int Offset;
	bool Word0;
	OperandRulePredicate(int offset, bool word0, bool optional): OperandRule(Predicate)
	{
		if(optional)
			Type = OperandType::Optional;
		Word0 = word0;
		Offset = offset;
	}
	virtual void Process(SubString &component)
	{
		unsigned int result;
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
		csCurrentInstruction.OpcodeWord0 &= ~(7<<Offset);
		if(Word0)
			csCurrentInstruction.OpcodeWord0 |= result;
		else
			csCurrentInstruction.OpcodeWord1 |= result;
	}
}OPRPredicate1(14, true, false), OPRPredicate0(17, true, false), OPRPredicate2NotNegatable(17, false, true);

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
		OPRRegister1.Process(component.SubStr(startPos, component.Length - startPos));
	}
}OPRFADD32IReg1;



#else
#define RulesOperandRegisterDefined
#endif