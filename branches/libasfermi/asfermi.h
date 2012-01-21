#ifndef ASFERMI_H
#define ASFERMI_H

#include <stack>
#include <list>
#include <iostream>
#include <fstream>
#include "DataTypes.h"
#include "Cubin.h"

struct Constant2Parser;

class ASFermi
{
	bool csSelfDebug;

	int csLineNumber;
	int csInstructionOffset;
	Line csCurrentLine;
	Instruction csCurrentInstruction;
	Directive csCurrentDirective;

	std::string ofilename;
	char *csSource;
	int csSourceSize;
	int csRegCount;
	int csBarCount;
	bool csAbsoluteAddressing;

	OperationMode csOperationMode;
	char* csSourceFilePath;

	bool csExceptionPrintUsage;
	bool csErrorPresent;

	//the following 3 variables are for Replace mode.
	int csOutputSectionOffset;
	int csOutputSectionSize;
	int csOutputInstructionOffset;

	// List of parsers to loaded at initialization.
	static std::list<MasterParser*> csMasterParserList;
	static std::list<LineParser*> csLineParserList;
	static std::list<InstructionParser*> csInstructionParserList;
	static std::list<DirectiveParser*> csDirectiveParserList;

	InstructionRule** csInstructionRules; //sorted array
	int* csInstructionRuleIndices; //Instruction name index of the corresponding element in csInstructionRules
	int csInstructionRuleCount;
	static std::list<InstructionRule*> csInstructionRulePrepList; //used for preperation

	DirectiveRule** csDirectiveRules; //sorted array
	int* csDirectiveRuleIndices; //Directive name index of the corresponding element in csDirectiveRules
	int csDirectiveRuleCount;
	static list<DirectiveRule*> csDirectiveRulePrepList; //used for preperation

	stack<MasterParser*> csMasterParserStack;
	stack<LineParser*> csLineParserStack;
	stack<InstructionParser*> csInstructionParserStack;
	stack<DirectiveParser*> csDirectiveParserStack;

	static MasterParser* csMasterParser;  //curent Master Parser
	static LineParser* csLineParser;
	static InstructionParser* csInstructionParser;
	static DirectiveParser* csDirectiveParser;

	list<Line> csLines;
	list<Instruction> csInstructions;
	list<Directive> csDirectives;
	list<Label> csLabels;
	list<LabelRequest> csLabelRequests;

	ELFSection cubinSectionEmpty, cubinSectionSHStrTab;
	ELFSection cubinSectionStrTab, cubinSectionSymTab;
	ELFSection cubinSectionConstant2, cubinSectionNVInfo;
	ELFSegmentHeader cubinSegmentHeaderPHTSelf;
	ELFSegmentHeader cubinSegmentHeaderConstant2;

	unsigned int cubinCurrentSectionIndex;
	unsigned int cubinCurrentOffsetFromFirst; //from the end of the end of .symtab
	unsigned int cubinCurrentSHStrTabOffset;
	unsigned int cubinCurrentStrTabOffset;
	unsigned int cubinTotalSectionCount;
	unsigned int cubinPHTOffset;
	unsigned int cubinPHCount;
	unsigned int cubinConstant2Size;
	unsigned int cubinCurrentConstant2Offset;
	bool cubinConstant2Overflown;

	void (*cubinCurrentConstant2Parser)(SubString &content);

	Architecture cubinArchitecture;
	bool cubin64Bit;

	bool csCurrentKernelOpened;
	Kernel csCurrentKernel;
	list<Kernel> csKernelList;

public :

	ASFermi();

	~ASFermi();
};

#endif // ASFERMI_H

