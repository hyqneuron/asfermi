#include "../DataTypes.h"
#include "../GlobalVariables.h"

#include "../stdafx.h"
//#include "stdafx.h" //SMark

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
			((OperandRule*)&OPRMOVStyle)->Process(component);
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
		if(component.Length>2 && component[0]=='0' && (component[1] == 'x' || component[1] == 'X'))
			result = component.ToImmediate32FromHexConstant(false);
		else
			result = component.ToImmediate32FromInt32();
		if(result>=32)
			throw 133;//shift can be no larger than 31
		csCurrentInstruction.OpcodeWord0 |= result << 5; //assumes that the opcode0 has unwritten field
	}
}OPRISCADDShift;

SubString *idx_NOPCCOP;
int *idx;
unsigned int *idx_type;

struct OperandRuleNOPCC: OperandRule
{
	bool Initialized;
	int computeIndex(SubString ss)
	{
		if(ss.Length>3)
			return -1;
		int result=0;
		int weight = 256*256;
		for(int i = 0; i<ss.Length; i++)
		{
			result += ss[i]*weight;
			weight/=256;
		}
		return result;
	}
	void Initialize()
	{
		Initialized = true;
		int size = 32;
		idx_NOPCCOP = new SubString[size];
		idx_type = new unsigned int[size];
		idx = new int[size];

		idx_NOPCCOP[0] ="F";
		idx_NOPCCOP[1] ="LT"; 
		idx_NOPCCOP[2] ="EQ"; 
		idx_NOPCCOP[3] ="LE"; 
		idx_NOPCCOP[4] ="GT"; 
		idx_NOPCCOP[5] ="NE"; 
		idx_NOPCCOP[6] ="GE"; 
		idx_NOPCCOP[7] ="NUM"; 
		idx_NOPCCOP[8] ="NAN"; 
		idx_NOPCCOP[9] ="LTU"; 
		idx_NOPCCOP[10]="EQU"; 
		idx_NOPCCOP[11]="LEU"; 
		idx_NOPCCOP[12]="GTU"; 
		idx_NOPCCOP[13]="NEU"; 
		idx_NOPCCOP[14]="GEU"; 
		idx_NOPCCOP[15]="T"; 
		idx_NOPCCOP[16]="OFF"; 
		idx_NOPCCOP[17]="LO"; 
		idx_NOPCCOP[18]="SFF"; 
		idx_NOPCCOP[19]="LS"; 
		idx_NOPCCOP[20]="HI"; 
		idx_NOPCCOP[21]="SFT"; 
		idx_NOPCCOP[22]="HS";
		idx_NOPCCOP[23]="OFT"; 
		idx_NOPCCOP[24]="CSM_TA"; 
		idx_NOPCCOP[25]="CSM_TR"; 
		idx_NOPCCOP[26]="CSM_MX"; 
		idx_NOPCCOP[27]="FCSM_TA"; 
		idx_NOPCCOP[28]="FCSM_TR"; 
		idx_NOPCCOP[29]="FCSM_MX"; 
		idx_NOPCCOP[30]="RLE"; 
		idx_NOPCCOP[31]="RGT"; 
		
		for(int i =0; i<size; i++)
		{
			idx[i] = computeIndex(idx_NOPCCOP[i]);
			idx_type[i] = (unsigned int)i;
		}
		delete[] idx_NOPCCOP;
		unsigned int typeSaver;
		int indexsaver;
		for(int i = size - 1; i >  0; i--)
		{
			for(int j =0; j< i; j++)
			{
				//larger one behind
				if(idx[j] > idx[j+1])
				{
					typeSaver = idx_type[j];
					indexsaver = idx[j];
					idx_type[j] = idx_type[j+1];
					idx[j] = idx[j+1];
					idx_type[j+1] = typeSaver;
					idx[j+1] = indexsaver;
				}
				else if(idx[j] == idx[j+1])
				{
					if(idx[j]!=-1)
						throw exception(); //repeating indices
				}
			}
		}
	}
	int findType(SubString ss)
	{
		int Index = computeIndex(ss);
		int start = 0; //inclusive
		int end = 32;//exclusive
		int mid;
		while(start<end) //still got unchecked numbers
		{
			mid = (start+end)/2;
			if(Index > idx[mid])
				start = mid + 1;
			else if(Index < idx[mid])
				end = mid;
			else
				return idx_type[mid];
		}
		return -1;
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

		unsigned int type = 0;

		if(mod.Length<=3)
		{
			type = findType(mod);
			if(type==-1)
				throw 135;
		}
		else
		{
			//FCSM_**
			if(mod[0]=='F')
			{
				type+=3;
				mod.Length--;
				mod.Start++;
			}
			//CSM_**
			if(mod.Length!=6 || !mod.SubStr(0, 4).Compare("CSM_"))
				throw 135;
			//TA & TR
			if(mod[4]=='T')
			{
				if(mod[5]=='A')
					type+=24;
				else if(mod[5]=='R')
					type+=25;
				else
					throw 135;
			}
			//MX
			else if(mod[4]=='M'&&mod[5]=='X')
				type+=26;
			else
				throw 135;
		}
		csCurrentInstruction.OpcodeWord0 &= ~(15<<5);
		csCurrentInstruction.OpcodeWord0 |= type<<5;
	}
	~OperandRuleNOPCC()
	{
		if(Initialized)
		{
			delete[] idx_type;
			delete[] idx;
		}
	}
}OPRNOPCC; //can only have one instance