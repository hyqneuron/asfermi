/*
This file contains basic data types used in asfermi.
0: SubString, defined in SubString.h
1: Basic structures used by the assembler: Component, Line, Instruction, Directive
2: Structures for line analysis: ModifierRule, OperandRule, InstructionRule
3: Abstract parser structures: Parser, MasterParser, LineParser, InstructionParser, DirectiveParser
*/

#ifndef DataTypesDefined //prevent multiple inclusion
//---code starts ---





using namespace std;

//-----Extern declarations
extern char* csSource;
extern int csMaxReg;
extern int hpParseComputeInstructionNameIndex(SubString &name);
extern int hpParseComputeDirectiveNameIndex(SubString &name);
//-----End of extern declaration

//	1
//-----Basic structures used by the assembler: Component, Line, Instruction, Directive

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
	bool Is8;	//true: OpcodeWord1 is used as well
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
		Components.clear();
		//not cleared: Is8, OpcodeWord, Predicated
	}
};
struct Directive
{
	SubString DirectiveString;
	int LineNumber;
	list<SubString> Parts;
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
		Parts.clear();
	}
};
//-----End of basic types





//	2
//-----Structures for line analysis: ModifierRule, OperandRule, InstructionRule, DirectiveRule
typedef enum OperandType
{
	Immediate32HexConstant, Immediate32IntConstant, Immediate32FloatConstant, Immediate32AnyConstant, 
	Register, GlobalMemoryWithImmediate32, ConstantMemory, SharedMemory, Optional, Custom, 
	MOVStyle, FADDStyle, IADDStyle
};
struct ModifierRule
{
	char* Name;
	int Length; //Length of name string

	bool Apply0; //apply on OpcodeWord0?
	unsigned int Mask0; // Does an AND operation with opcode first
	unsigned int Bits0; //then an OR operation

	bool Apply1; //Apply on OpcodeWord1?
	unsigned int Mask1;
	unsigned int Bits1;

	bool NeedCustomProcessing;
	virtual void CustomProcess(Component &component){}
	ModifierRule(){}
	ModifierRule(char* name, int length, bool apply0, bool apply1, bool needCustomProcessing)
	{
		Name = name;
		Length = length;
		Apply0 = apply0;
		Apply1 = apply1;
		NeedCustomProcessing = needCustomProcessing;
	}
};
struct OperandRule
{
	OperandType Type;
	int ModifierCount;
	ModifierRule** ModifierRules;

	OperandRule(){}
	OperandRule(OperandType type, int modifierCount)
	{
		Type = type;
		ModifierCount = modifierCount;
		if(ModifierCount!=0)
			ModifierRules = new ModifierRule*[ModifierCount];
	}
	virtual void Process(Component &component) = 0;
	~OperandRule()
	{
		if(ModifierCount!=0)
			delete[] ModifierRules;
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
	virtual void CustomProcess(){}
	int ComputeIndex()
	{
		int result = 0;
		int len = strlen(Name);
		SubString nameString;
		nameString.Start = Name;
		nameString.Length = len;
		result = hpParseComputeInstructionNameIndex(nameString);
		return result;
	}
	InstructionRule(){};
	InstructionRule(char* name, int operandCount, int modifierCount, bool is8, bool needCustomProcessing)
	{
		Name = name;
		OperandCount = operandCount;
		ModifierCount = modifierCount;
		if(operandCount>0)
			Operands = new OperandRule*[operandCount];
		if(modifierCount>0)
			ModifierRules = new ModifierRule*[modifierCount];
		Is8 = is8;
		NeedCustomProcessing = needCustomProcessing;
	}
	~InstructionRule()
	{
		if(OperandCount>0)
			delete[] Operands;
		if(ModifierCount>0)
			delete[] ModifierRules;
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

struct DirectiveRule
{
	char* Name;

	virtual void Process() = 0;
	int ComputeIndex()
	{
		int result = 0;
		int len = strlen(Name);
		SubString nameString;
		nameString.Start = Name;
		nameString.Length = len;
		result = hpParseComputeDirectiveNameIndex(nameString);
		return result;
	}
};
//-----End of structures for line analysis






//	3
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
	virtual void Parse() = 0;
};
struct DirectiveParser: Parser
{
	virtual void Parse() = 0;
};
//-----End of abstract parser structures








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
#else
#define DataTypesDefined
#endif