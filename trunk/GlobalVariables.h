//prevent multiple inclusion
#if defined GlobalVariablesDefined 
#else
#define GlobalVariablesDefined yes

//---code starts ---
#include <vld.h>

#include <iostream>
#include <fstream>
#include <list>
#include "DataTypes.h"

using namespace std;

//All globla variables are prefixed with cs, meaning "current state"
//-----Global variables

int csLineNumber = 0;
int csInstructionOffset;

fstream csInput;
fstream csOutput;
char *csSource;
int csMaxReg = 0;

bool csReplaceMode;
bool csExceptionPrintUsage = false;
bool csErrorPresent = false;
int csOutputSectionOffset;
int csOutputSectionSize;
int csOutputInstructionOffset;

list<MasterParser*>		csMasterParserList;  //List of parsers to loaded at initialization
list<LineParser*>		csLineParserList;
list<InstructionParser*> csInstructionParserList;
list<DirectiveParser*>	csDirectiveParserList;


InstructionRule** csInstructionRules; //sorted array
int* csInstructionRuleIndices; //Instruction name index of the corresponding element in csInstructionRules
int csInstructionRuleCount;
list<InstructionRule*>  csInstructionRulePrepList; //used for preperation

MasterParser*		csMasterParser;  //curent Master Parser
LineParser*			csLineParser;
InstructionParser*	csInstructionParser;
DirectiveParser*	csDirectiveParser;

list<Line> csLines;
list<Instruction> csInstructions;
list<Directive> csDirectives;
list<LabelRequest> csLabelRequests;
list<Label> csLabels;

//-----End of global variables

#endif