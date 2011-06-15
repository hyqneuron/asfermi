#if defined DataTypesDefined //prevent multiple inclusion
#else
#define DataTypesDefined yes
//---code starts ---
//#include <vld.h> //remove when you compile



#include <string.h>
#include <list>
#include "SubString.h"

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
		if(len<1)return 0;
		result += (int)Name[0] * 2851;
		if(len<2) return result;
		result += (int)Name[1] * 349;
		for(int i =2; i<len; i++)
			result += (int)Name[i];
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
	virtual void Parse() = 0;
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
	void Parse();
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