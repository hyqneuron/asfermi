/*
This file contains rules for operands
*/
#ifndef RulesOperandDefined

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





//	5.2
//-----Specific Operand Rules
/*
Primary operand types: 
	Type1:	Register, Hex Constant, Predicate Register
	Type2:	Integer Constant, Float Constant, S2R operand
	Type3:	Constant Memory, Global Memory, Constant Memory
	
Primary operand types have their range checking done in SubString member functions

Secondary operand types:
	MOVStyle	:Register, Constant memory without reg, 20-bit Hex, Int, Float)
	FADDStyle	:Register, Constant memory without reg, 20-bit float)
	FIADDStyle	:Register, Constant memory without reg, 20-bit Int)

Secondary operand processors only check that components have the correct prefix and then send components to SubString
	member functions for processing. Primary operand processors then directly write the result to OpcodeWord.
Composite operator processor would sometimes have to check the value returned by SubString functions fall within
	a defined, smaller range before writing.
*/
//SubString member functions only assume a least length of 1



//---Type1 Operand Rules
//Register Operand
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

struct OperandRuleImmediate32HexConstant: OperandRule
{
	OperandRuleImmediate32HexConstant() :OperandRule(Immediate32HexConstant){}
	virtual void Process(SubString &component)
	{
		unsigned int result = component.ToImmediate32FromHexConstant(true);
		WriteToImmediate32(result);
	}	
}OPRImmediate32HexConstant;

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
//---End of type1 operand rules


//---Type2 operand rules
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
//---End of type2 operand rules







//Type3 operand rules
//Global Memory Operand
struct OperandRuleGlobalMemoryWithImmediate32: OperandRule
{
	OperandRuleGlobalMemoryWithImmediate32(): OperandRule(GlobalMemoryWithImmediate32){}
	virtual void Process(SubString &component)
	{
		unsigned int memory; int register1;
		component.ToGlobalMemory(register1, memory);
		//Check max reg when register is not RZ(63)
		if(register1!=63)
			csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		WriteToImmediate32(memory);
	}
}OPRGlobalMemoryWithImmediate32;

struct OperandRuleSharedMemoryWithImmediate20: OperandRule
{
	OperandRuleSharedMemoryWithImmediate20(): OperandRule(SharedMemoryWithImmediate20){}
	virtual void Process(SubString &component)
	{
		unsigned int memory; int register1;
		component.ToGlobalMemory(register1, memory);
		if(memory>=1<<20) //issue: not sure if negative hex is gonna work
			throw 130; //cannot be longer than 20 bits
		//Check max reg when register is not RZ(63)
		if(register1!=63)
			csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		WriteToImmediate32(memory);
	}
}OPRSharedMemoryWithImmediate20;



//Constant Memory Operand
struct OperandRuleConstantMemory: OperandRule
{
	OperandRuleConstantMemory() : OperandRule(ConstantMemory){}
	virtual void Process(SubString &component)
	{		
		unsigned int bank, memory;
		int register1;
		component.ToConstantMemory(bank, register1, memory);
		if(register1!=63)
			csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		csCurrentInstruction.OpcodeWord1 |= bank<<10;
		WriteToImmediate32(memory);
		//no need to do the marking for constant memory
	}
}OPRConstantMemory;
//---End of type3 operand rules







//ignored operand: currently used for NOP
struct OperandRuleIgnored: OperandRule
{
	OperandRuleIgnored() : OperandRule(OperandType::Optional){}
	virtual void Process(SubString &component)
	{
		//do nothing
	}
}OPRIgnored;









//---Secondary operands rules
//MOV: Register, Constant memory without reg, 20-bit Hex)
struct OperandRuleMOVStyle: OperandRule
{
	OperandRuleMOVStyle() : OperandRule(MOVStyle){}

	virtual void Process(SubString &component)
	{
		//Register
		if(component[0] == 'R' || component[0] == 'r')
			OPRRegister2.Process(component);
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


#define mArithmeticCommonEnd {if(negate){component.Start--;	component.Length++;	}}

#define mArithmeticCommon\
	bool negate = false;\
	if(component[0] == '-')\
	{\
		if(!AllowNegate)\
			throw 129;\
		negate = true;\
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
	bool AllowNegate;
	OperandRuleFADDStyle(bool allowNegate) :OperandRule(FADDStyle)
	{
		AllowNegate = allowNegate;
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
	bool AllowNegate;
	OperandRuleIADDStyle(bool allowNegate) :OperandRule(IADDStyle)
	{
		AllowNegate = allowNegate;
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

struct OperandRuleLOP: OperandRule
{
	int ModShift;
	OperandRuleLOP(int modShift): OperandRule(Custom)
	{
		ModShift = modShift;
	}
	virtual void Process(SubString &component)
	{
		bool not = false;
		if(component[0]=='~')
		{
			not = true;
			component.Start++;
			component.Length--;
		}
		if(component.Length<1)
			throw 132; //empty operand
		if(ModShift==8)
			OPRMOVStyle.Process(component);
		else
			OPRRegister1.Process(component);
		if(not)
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
			OPRFMULStyle.Process(component);
		else
			OPRIMULStyle.Process(component);
		if(operated)
		{
			component.Start--;
			component.Length++;
		}
	}
}OPRF2I(true), OPRI2F(false);


//---End of secondary operand rules
//-----End of Specifc Operand Rules




#else
#define RulesOperandDefined
#endif