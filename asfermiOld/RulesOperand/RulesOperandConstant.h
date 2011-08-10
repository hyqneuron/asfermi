#ifndef RulesOperandConstantDefined





//24-bit Hexadecimal Constant Operand
struct OperandRuleImmediate24HexConstant: OperandRule
{
	OperandRuleImmediate24HexConstant() :OperandRule(Immediate32HexConstant){} //issue: wrong type
	virtual void Process(SubString &component)
	{
		bool negate = false;
		if(component[0]=='-')
		{
			negate = true;
			component.Start++;
			component.Length--;
		}
		unsigned int result = component.ToImmediate32FromHexConstant(false);
		if(negate)
		{
			if(result>0x7FFFFF)
				throw 131; //too large offset
			result ^= 0xFFFFFF;
			result += 1;
			component.Start--;
			component.Length++;
		}
		else
		{
			if(result>0xFFFFFF)
				throw 131;
		}
		WriteToImmediate32(result);
	}	
}OPRImmediate24HexConstant;

/*
struct OperandRuleImmediateHexConstant: OperandRule
{
	unsigned int MaxNum;
	OperandRuleImmediateHexConstant(int maxBit)
	{
		MaxNum = 1<<maxBit - 1;
	}
	virtual void Process(SubString &component)
	{
		bool negate = false;		
		if(component[0]=='-')
		{
			negate = true;
			component.Start++;
			component.Length--;
		}
		unsigned int result = component.ToImmediate32FromHexConstant(false);
		
		if(!negate)
		{
			if(result > MaxNum)
				throw 131;
		}
		else
		{
			if(result>MaxNum>>1)
				throw 131;
		}
	}
};
*/

struct OperandRuleImmediate32HexConstant: OperandRule
{
	OperandRuleImmediate32HexConstant() :OperandRule(Immediate32HexConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromHexConstant(true);
		WriteToImmediate32(result);
	}	
}OPRImmediate32HexConstant;



//32-bit Integer Constant Operand
struct OperandRuleImmediate32IntConstant: OperandRule
{
	OperandRuleImmediate32IntConstant():OperandRule(Immediate32IntConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromIntConstant();
		WriteToImmediate32(result);
	}
}OPRImmediate32IntConstant;

//32-bit Floating Number Constant Operand
struct OperandRuleImmediate32FloatConstant: OperandRule
{
	OperandRuleImmediate32FloatConstant():OperandRule(Immediate32FloatConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromFloatConstant();
		WriteToImmediate32(result);
	}	
}OPRImmediate32FloatConstant;

//32-bit Constant: Hex || Int || Float
struct OperandRuleImmediate32AnyConstant: OperandRule
{
	OperandRuleImmediate32AnyConstant():OperandRule(Immediate32AnyConstant){}
	virtual void Process(SubString &component)
	{
		//Issue: not yet implemented
	}	
}OPRImmediate32AnyConstant;
struct OperandRuleS2R: OperandRule
{
	OperandRuleS2R(): OperandRule(OperandType::Custom){}
	virtual void Process(SubString &component)
	{
		unsigned int result;
		if(component.Length<8 || component[0]!='S' || component[1]!='R' || component[2]!='_')
			throw 128; //incorrect special register name
		//WarpID
		if(component[3]=='W')
		{
			if(!component.Compare(SubString("SR_WarpID")))
				throw 128;
			result = 0;
		}
		//GridParam
		else if(component[3]=='G')
		{
			if(!component.Compare(SubString("SR_GridParam")))
				throw 128;
			result = 0x2c;
		}
		else if(component[3]=='V')
		{			
			if(component.Compare(SubString("SR_VirtId")))
				result = 3;
			else if(component.Compare(SubString("SR_VirtCfg")))
				result = 2;
			else throw 128;
		}
		//Tid
		else if(component[3]=='T')
		{
			if(component.Compare(SubString("SR_Tid_X")))
				result = 0x21;
			else if(component.Compare(SubString("SR_Tid_Y")))
				result = 0x22;
			else if(component.Compare(SubString("SR_Tid_Z")))
				result = 0x23;
			else throw 128;
		}
		//CTAID/Clock
		else if(component[3]=='C')
		{
			if(component.Length==10)
			{
				if(component.Compare(SubString("SR_CTAid_X")))
					result = 0x25;
				else if(component.Compare(SubString("SR_CTAid_Y")))
					result = 0x26;
				else if(component.Compare(SubString("SR_CTAid_Z")))
					result = 0x27;
				else throw 128;
			}
			else
			{
				if(component.Compare(SubString("SR_Clock_Hi")))
					result = 0x11;
				else if(component.Compare(SubString("SR_Clock_Lo")))
					result = 0x10;
				else throw 128;
				csCurrentInstruction.OpcodeWord1 |= 1;
			}
		}
		else
			throw 128;
		csCurrentInstruction.OpcodeWord0 |= result << 26;
	}
}OPRS2R;

#else
#define RulesOperandConstantDefined
#endif