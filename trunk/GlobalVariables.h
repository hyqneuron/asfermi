/*
This file contains global variables used throughout the assembler.
All globla variables are prefixed with 'cs', meaning "current state"
*/

#ifndef GlobalVariablesDefined //prevent multiple inclusion
//---code starts ---


using namespace std;

extern struct Constant2Parser;

bool csSelfDebug = false;

int csLineNumber = 0;
int csInstructionOffset;
Line		csCurrentLine;
Instruction csCurrentInstruction;
Directive   csCurrentDirective;

fstream csInput;
fstream csOutput;
char *csSource;
int csSourceSize;
int csMaxReg = 0;

enum OperationMode{Replace, Insert, DirectOutput, Undefined };
OperationMode csOperationMode = Undefined;
char* csSourceFilePath;

bool csExceptionPrintUsage = false;
bool csErrorPresent = false;

//the following 3 variables are for Replace mode.
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

DirectiveRule** csDirectiveRules; //sorted array
int* csDirectiveRuleIndices; //Directive name index of the corresponding element in csDirectiveRules
int csDirectiveRuleCount;
list<DirectiveRule*>  csDirectiveRulePrepList; //used for preperation

stack<MasterParser*>	csMasterParserStack;
stack<LineParser*>		csLineParserStack;
stack<InstructionParser*>  csInstructionParserStack;
stack<DirectiveParser*> csDirectiveParserStack;

MasterParser*		csMasterParser;  //curent Master Parser
LineParser*			csLineParser;
InstructionParser*	csInstructionParser;
DirectiveParser*	csDirectiveParser;

list<Line> csLines;
list<Instruction> csInstructions;
list<Directive> csDirectives;
list<LabelRequest> csLabelRequests;
list<Label> csLabels;

//=================

ELFSection cubinSectionEmpty, cubinSectionSHStrTab, cubinSectionStrTab, cubinSectionSymTab;
ELFSection cubinSectionConstant2, cubinSectionNVInfo;
ELFSegmentHeader cubinSegmentHeaderPHTSelf;
ELFSegmentHeader cubinSegmentHeaderConstant2;


unsigned int cubinCurrentSectionIndex = 0;
unsigned int cubinCurrentOffsetFromFirst = 0; //from the end of the end of .symtab
unsigned int cubinCurrentSHStrTabOffset = 0;
unsigned int cubinCurrentStrTabOffset = 0;
unsigned int cubinTotalSectionCount =0;
unsigned int cubinPHTOffset = 0;
unsigned int cubinPHCount;
unsigned int cubinConstant2Size = 0;
unsigned int cubinCurrentConstant2Offset = 0;
bool cubinConstant2Overflown = false;

void (*cubinCurrentConstant2Parser)(SubString &content);

enum Architecture{sm_20, sm_21};
Architecture cubinArchitecture = sm_20; //default architecture is sm_20
bool cubin64Bit = false;



char *cubin_str_empty = "";
char *cubin_str_shstrtab =	".shstrtab";
char *cubin_str_strtab =	".strtab";
char *cubin_str_symtab =	".symtab";
char *cubin_str_extra1 =	".nv.global.init";
char *cubin_str_extra2 =	".nv.global";
char *cubin_str_text   =	".text.";
char *cubin_str_constant0=	".nv.constant0.";
char *cubin_str_info   =	".nv.info.";
char *cubin_str_shared =	".nv.shared.";
char *cubin_str_local  =	".nv.local.";
char *cubin_str_constant2=	".nv.constant2";
char *cubin_str_nvinfo =	".nv.info";


bool csCurrentKernelOpened = false;
Kernel	csCurrentKernel;
list<Kernel> csKernelList;


#else
#define GlobalVariablesDefined
#endif