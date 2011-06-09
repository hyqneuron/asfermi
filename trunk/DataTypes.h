#if defined DataTypesDefined //prevent multiple inclusion
#else
#define DataTypesDefined yes
//---code starts ---



using namespace std;

//-----Forward, extern declarations
struct Instruction;
struct Directive;
struct InstructionParser;
struct DirectiveParser;
extern list<Instruction> csInstructions;
extern list<Directive> csDirectives;
extern InstructionParser *csInstructionParser;
extern DirectiveParser *csDirectiveParser;
extern char* csSource;
extern int csMaxReg;
//-----End of forward declaration

//	1.0
//-----Basic structures used by the assembler: SubString, Component, Line, Instruction, Directive
int d_currentPos; //used by various functions in SubString
static const unsigned int op_HexRef[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
struct SubString
{
	int Offset;	//global offset in csSource
	int Length;
	char* Start;
	SubString(){}
	SubString(int offset, int length)
	{
		Offset = offset;
		Length = length;
		Start = csSource + offset;
		if(length<0)
			throw exception();
	}
	char operator [] (int position)
	{
		return Start[position];
	}
	int Find(char target, int startPos) //startPos is the position in this string
	{
		for(d_currentPos = startPos; d_currentPos < Length; d_currentPos++)
		{
			if(target == Start[d_currentPos])
				return d_currentPos;
		}
		return -1;
	}
	int FindInSource(char target, int startPos, int &lengthFromStartPos) //startPos is the position in this string. lengthFromStart is the d_currentPos - startPos
	{
		for(d_currentPos = startPos; d_currentPos < Length; d_currentPos++)
		{
			if(target == Start[d_currentPos])
			{
				lengthFromStartPos = d_currentPos - startPos;
				return d_currentPos + Offset;
			}
		}
		return -1;
	}
	int FindBlank(int startPos)
	{
		for(d_currentPos = startPos; d_currentPos < Length; d_currentPos++)
		{
			if(Start[d_currentPos] < 33)
				return d_currentPos;
		}
		return -1;
	}
	SubString SubStr(int startPos, int length)
	{
		SubString result(startPos + Offset, length);
		if(length<0)
			throw exception();
		return result;
	}
	char* ToCharArray()
	{
		char *result = new char[Length + 1];
		for(int i =0 ; i<Length; i++)
			result[i] = Start[i];
		result[Length] = (char)0;
		return result;
	}
	bool CompareWithCharArrayIgnoreEndingBlank(char* target, int length) //length is the length of the char
	{
		if(length < Length)
			return false;
		for(int i =0; i<length; i++)
		{
			if(Start[i]!=target[i])
				return false;
		}
		for(int i = Length; Length < length; i++) //it's fine for the target to end with blanks
		{
			if(target[i]>32)
				return false;
		}
		return true;
	}
	int ToRegister()
	{
		int result;
		if(Length<2 || Start[0]!='R')
		{
			throw 55; //Incorrect register
			//return;
		}
		if(Start[1] == 'Z')
		{
			return 63;
		}
		result = (int)Start[1] - 48;
		if(result<0 || result>9)
		{
			throw 55; //incorrect register
			//return;
		}
		if(Length==2)
		{
			return result;
		}
		int n2 = (int)Start[2] - 48;
		if(n2<0 || n2>9)
		{
			return result;
		}
		result *= 10;
		result += n2;
		if(result>=63)
		{
			throw 56; //register number too large
			//return;
		}
		return result;
	}
		
#define mReduceWithError(o,error)				\
{												\
	if(o<58)									\
	{											\
		if(o<48)								\
		{										\
			result >>= shift + 4;				\
			break;								\
		}										\
		o-=48;									\
	}											\
	else if(o<71)								\
	{											\
		if(o<65)								\
		{										\
			result >>= shift + 4;				\
			break;								\
		}										\
		o-=55;									\
	}											\
	else if(o<103)								\
	{											\
		if(o<97)								\
		{										\
			result >>= shift + 4;				\
			break;								\
		}										\
		o -= 87;								\
	}											\
	else										\
	{											\
		result >>= shift + 4;					\
		break;									\
	}											\
}
	
	unsigned int ToImmediate32FromHex()
	{
		if(Length<3 || Start[0]!='0' || (Start[1]!='x'&& Start[1] != 'X') )
		{
			throw 57; //incorrect hex
			//return;
		}
		unsigned int  result = 0;
		int maxlength = (Length<10)? Length:10;
		int shift = Length * 4 -12;
		int digit;
		int i;
		for(i =2; i<maxlength; i++)
		{
			digit = (int) Start[i];
			mReduceWithError(digit, 58); //58: invalid hex constant
			result |= op_HexRef[digit] << shift;
			shift -= 4;
		}
		if(i==2)
		{
			throw 57;
			//return;
		}
		return result;
	}
	void ToGlobalMemory(int &register1, unsigned int&memory)
	{
		register1 = 63; //RZ
		memory = 0;
		if(Length < 4 || Start[0]!='[') //[R0]
		{
			throw 59; //incorrect global mem
			//return;
		}
		int startPos = 1;
		while(startPos<Length)
		{
			if(Start[startPos] > 32)
				break;
			startPos++;
		}
		int plusPos = Find('+', 0);
		if(plusPos==-1)
		{
			if(Start[1]=='R')
				register1 = (SubStr(startPos, Length -startPos)).ToRegister();
			else
				memory = SubStr(startPos, Length -startPos ).ToImmediate32FromHex();
		}
		else
		{
			register1 = SubStr(startPos, Length -startPos).ToRegister();
			startPos = plusPos+1;
			while(startPos<Length)
			{
				if(Start[startPos] > 32)
					break;
				startPos++;
			}
			memory = SubStr(startPos, Length - startPos).ToImmediate32FromHex();
		}
	}
};
struct Component //Component can be either an instruction name or an operand
{
	SubString Content; //instruction name or operand name, without modifier
	list<SubString> Modifiers;
	Component(){}
	Component(SubString content)
	{
		Content = content;
	}
};
struct Line
{
	SubString LineString;
	int LineNumber;
	Line(){}
	Line(SubString lineString, int lineNumber)
	{
		LineString = lineString;
		LineNumber = lineNumber;
	}
};
struct Instruction
{
	SubString InstructionString;
	int LineNumber;
	list<Component> Components;	//eg: LDS.128 R0, [0x0]; has 3 components: LDS.128, R0 and [0x0]. The first component has 1 Modifier: 128. The first component would be the instruction name
	bool Is8;	//true: Opcode8; false: Opcode4
	unsigned int OpcodeWord0;
	unsigned int OpcodeWord1;
	int Offset;	//Instruction offset in assembly
	bool Predicated;
	
	Instruction(){}
	Instruction(SubString instructionString, int offset, int lineNumber)
	{
		InstructionString = instructionString;
		Offset = offset;
		LineNumber = lineNumber;
	}
	void Reset(SubString instructionString, int offset, int lineNumber)
	{
		InstructionString = instructionString;
		Offset = offset;
		LineNumber = lineNumber;
	}
};
struct Directive
{
	SubString DirectiveString;
	int LineNumber;
	Directive(){}
	Directive(SubString directiveString, int lineNumber)
	{
		DirectiveString = directiveString;
		LineNumber = lineNumber;
	}
	void Reset(SubString directiveString, int lineNumber)
	{
		DirectiveString = directiveString;
		LineNumber = lineNumber;
	}
};
//-----End of basic types





//	2.0
//-----Structures for instruction analysis: ModifierRule, OperandRule, InstructionRule
typedef enum OperandType
{
	Register, Immediate32, GlobalMemory, ConstantMemory, SharedMemory, Optional, Others
};
struct ModifierRule
{
	char* Name;
	int Length; //Length of name string

	bool Apply0;
	unsigned int Mask0;
	unsigned int Bits0;
	bool Apply1;
	unsigned int Mask1;
	unsigned int Bits1;

	bool NeedCustomProcessing;
	virtual void CustomProcess(Instruction &instruction,Component &component){}
	~ModifierRule()
	{
		//delete Name;
	}
};
struct OperandRule
{
	OperandType Type;
	int ModifierCount;
	ModifierRule* ModifierRules;

	virtual void Process(Instruction &instruction, Component &component) = 0;
	~OperandRule()
	{
		//delete[] ModifierRules;
	}
};
//When an instruction rule is initialized, the ComputeIndex needs to be called. They need to be sorted according to their indices and then placed in csInstructionRules;
struct InstructionRule
{
	char* Name;
	int OperandCount;
	OperandRule** Operands;
	int ModifierCount;
	ModifierRule** ModifierRules;

	bool Is8;
	unsigned int OpcodeWord0;
	unsigned int OpcodeWord1;

	bool NeedCustomProcessing;
	virtual void CustomProcess(Instruction &instruction){}
	int ComputeIndex()
	{
		int result = 0;
		int len = strlen(Name);
		if(len<1)return 0;
		result += (int)Name[0] * 2851;
		if(len<2) return result;
		result += (int)Name[1] * 349;
		for(int i =2; i<len; i++)
			result += (int)Name[i];
		return result;
	}
	InstructionRule(){};
	void Set(char* name, int operandCount, int modifierCount, bool is8)
	{
		Name = name;
		OperandCount = operandCount;
		ModifierCount = modifierCount;
		Operands = new OperandRule*[operandCount];
		Is8 = is8;
	}
	~InstructionRule()
	{
		//delete Name;
		//delete[] Operands;
		//delete[] ModifierRules;
	}
	static void BinaryStringToOpcode4(char* string, unsigned int &word0) //little endian
	{
		word0 = 0;
		for(int i =0; i<32; i++)
		{
			if(string[i]=='1')
				word0 |=  1u<<i;
		}
	}
	static void BinaryStringToOpcode8(char* string, unsigned int &word0, unsigned int &word1)
	{
		word0 = 0;
		for(int i =0; i<32; i++)
		{
			if(string[i]=='1')
				word0 |=  1<<i;
		}
		for(int i =0; i< 32; i++)
		{
			if(string[i+32]=='1')
				word1 |= 1<<i;
		}
	}
};
//-----End of structures for instruction analysis






//	3.0
//-----Abstract parser structures: Parser, MasterParser, LineParser, InstructionParser, DirectiveParser
struct Parser
{
	char* Name;
};
struct MasterParser: Parser
{
	virtual void Parse(unsigned int startinglinenumber) = 0;
};
struct LineParser: Parser
{
	virtual void Parse(Line &line) = 0;
};
struct InstructionParser: Parser
{
	virtual void Parse(Instruction &instruction) = 0;
};
struct DirectiveParser: Parser
{
	virtual void Parse(Directive &directive) = 0;
};
//-----End of abstract parser structures





//	4.0
//-----Default parsers
//Implementation of the various parse functions are in asfermi.cpp
struct DefaultMasterParser: MasterParser
{
	DefaultMasterParser()
	{
		Name = "DefaultMasterParser";
	}
	void Parse(unsigned int startinglinenumber);
};
struct DefaultLineParser : LineParser
{
	DefaultLineParser()
	{
		Name = "DefaultLineParser";
	}
	void Parse(Line &line);
};
struct DefaultInstructionParser: InstructionParser
{
	DefaultInstructionParser()
	{
		Name = "DefaultInstructionParser";
	}
	void Parse(Instruction &instruction);
};
struct DefaultDirectiveParser: DirectiveParser
{
	DefaultDirectiveParser()
	{
		Name = "DefaultDirectiveParser";
	}
	void Parse(Directive &directive);
};
//-----End of default parsers


//	9.0
//-----Label structures
struct Label
{
	string Name;
	int Offset;
};
struct LabelRequest
{
	Instruction *RelatedInstruction;
	string RequestedLabelName;
};
//-----End of Label structures


#endif