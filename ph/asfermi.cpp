using namespace std;



#include <stdarg.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <list>
#include <stack>

#include "SubString.h"
#include "DataTypes.h"
#include "Cubin.h"
#include "GlobalVariables.h"
#include "helper.h"
#include "SpecificParsers.h"

#include "stdafx.h"

#include "RulesModifier.h"
#include "RulesOperand.h"
#include "RulesInstruction.h"
#include "RulesDirective.h"





//-----Forward declarations
void ProcessCommandsAndReadSource(int argc, char** args);
void Initialize();
void WriteToCubinReplace();
void WriteToCubinDirectOutput();
//extern void ExternInitialize();
//-----End of forward declarations


int main(int argc, char** args)
{
	try
	{
		//---Preprocess
		ProcessCommandsAndReadSource(argc, args);
		Initialize();	//Initialize instruction and directive rules
		//ExternInitialize(); //initialize custom rules


		//---Process
		csMasterParser->Parse(0); //starting from line 0


		//---Postprocess
		if(csErrorPresent)
			throw 98; //cannot proceed due to errors
		
		if(csOperationMode == Replace)
			WriteToCubinReplace();
		else if(csOperationMode == DirectOutput)
			WriteToCubinDirectOutput();
		else
			throw 97; //Mode not supported
		puts("Done");
	}
	catch(int e)
	{
		hpExceptionHandler(e);
		hpCleanUp();
		return -1;
	}

	//---Normal exit
	hpCleanUp();
	getchar();
	return 0;
}

void WriteToCubinReplace()
{
	if(csInstructionOffset + csOutputInstructionOffset> csOutputSectionSize )
		throw 9; //output section not large enough
	list<Instruction>::iterator inst = csInstructions.begin();
	csOutput.seekp(csOutput.beg + ::csOutputSectionOffset + ::csOutputInstructionOffset);

	while(inst != csInstructions.end())
	{
		csOutput.write((char*)&inst->OpcodeWord0, 4);
		if(inst->Is8)
			csOutput.write((char*)&inst->OpcodeWord1, 4);
		inst++;
	}
	csOutput.flush();
}



void WriteToCubinDirectOutput()
{
	if(csInstructions.size()!=0)
		throw 100; //last kernel not ended
	if(csKernelList.size()==0)
		throw 101; //no valid kernel found
	


	
	//head sections: (null), .shstrtab, .strtab, .symtab
	//kern sections: .text.kername, .nv.constant0.kername, (not implemented).nv.constant16.kername,
	//				 .nv.info.kername, (optional).nv.shared.kername, (optional).nv.local.kername
	//tail sections: .nv.constant2, .nv.info

	/*
	Stage1: Set section index and section name for all sections
	1. head sections: SectionIndex, SHStrTabOffset
	2. kern sections: SectionIndex, SHStrTabOffset, StrOffset of .text
	3. tail sections: SectionIndex, SHStrTabOffset
	4. Confirm sizes of .shstrtab and .shstr and total section count
	*/
	hpCubinStage1();

	/*
	Stage2: Set SectionSize for all kernels
	1. head sections: SectionSize, SectionContent
		.symtab: Set SymbolIndex for all sections; set GlobalSymbolIndex for all kernels
	*/
	hpCubinStage2();

	/*
	Stage3
	1. kern sections: SectionSize, SectionContent
	2. tail sections: SectionSize, SectionContent
	*/
	hpCubinStage3();

	//Stage4: Setup all section headers
	hpCubinStage4();
	//Stage5: Setup all program segments
	hpCubinStage5();
	//Stage6: Setup ELF header
	hpCubinStage6();
	//Stage7: Write to cubin
	hpCubinStage7();




}

void ProcessCommandsAndReadSource(int argc, char** args)
{
	if(argc<4) //at least 4 strings: progpath inputpath -o outputname
	{
		csExceptionPrintUsage = true;
		throw 20; // invalid arguments
	}
	int currentArg = 1; //used to refer the next argument to be examined
	
	//Check input state. Single line instruction or input file?
	if(strcmp(args[1], "-I") == 0) // Single line instruction. 
	{
		//At least 7 args: progpath -I "instruction" -r target kernelname offset
		if(argc<7)
		{
			csExceptionPrintUsage = true;
			throw 20;
		}
		csSource = args[2]; //instruction string
		csLines.push_back(Line(SubString(0, strlen(csSource)), 0)); //directly create a single line, issue: single-line mode doesn't support comment
		currentArg = 3;
		
		if(strcmp(args[3], "-r")!=0)
			throw 10; //Single-line mode only supported in Replace Mode.
	}
	else //progpath inputpath ...
	{
		hpReadSource(args[1]);	//Call helper function to read input file and create lines
		currentArg = 2;
	}

	//Check options
	while(currentArg<argc)
	{
		if(strcmp(args[currentArg], "-r")==0)	//replace mode
		{
			if(argc - currentArg < 4)
			{
				csExceptionPrintUsage = true;
				throw 20; // invalid arguments
			}
			csOperationMode = Replace;
			hpCheckOutputForReplace(args[currentArg+1], args[currentArg+2], args[currentArg+3]); //Use helper function to check validity of input arguments
							//csOutput is opened without closing 
			currentArg += 4;
		}
		else if(strcmp(args[currentArg], "-o")==0) //direct output mode
		{
			if(argc - currentArg < 2)
			{
				csExceptionPrintUsage = true;
				throw 20; // invalid arguments
			}
			csOperationMode = DirectOutput;
			csOutput.open(args[currentArg + 1], fstream::out | fstream::binary |fstream::trunc);
			if(!csOutput.is_open() || !csOutput.good())
				throw 4; //failed to open output file
			currentArg += 2;
			//note that csOutput is not closed
		}
		else if(strcmp(args[currentArg], "-sm_20")==0)
		{
			cubinArchitecture = sm_20;
			currentArg += 1;
		}
		else if(strcmp(args[currentArg], "-sm_21")==0)
		{
			cubinArchitecture = sm_21;
			currentArg += 1;
		}
		/*
		else if(strcmp(args[currentArg], "-32")==0)
		{
			cubin64Bit = false;
			currentArg += 1;
		}
		else if(strcmp(args[currentArg], "-64")==0)
		{
			cubin64Bit = true;
			currentArg += 1;
		}
		*/
		else if(strcmp(args[currentArg], "-SelfDebug")==0)
		{
			csSelfDebug = true;
			currentArg += 1;
		}
		else
		{
			csExceptionPrintUsage = true;
			throw 0; //invalid argument
		}
	}

}

void OrganiseRules()
{
	int size = csInstructionRulePrepList.size();
	csInstructionRuleCount = size;
	csInstructionRules = new InstructionRule*[size];
	csInstructionRuleIndices = new int[size];
	int index = 0;
	for(list<InstructionRule*>::iterator rule = csInstructionRulePrepList.begin(); rule != csInstructionRulePrepList.end(); rule++)
	{
		csInstructionRules[index] = *rule;
		csInstructionRuleIndices[index] = (*rule)->ComputeIndex();
		index++;
	}
	InstructionRule *instSaver;
	int indexsaver;
	//sort the indices
	for(int i = size - 1; i >  0; i--)
	{
		for(int j =0; j< i; j++)
		{
			//larger one behind
			if(csInstructionRuleIndices[j] > csInstructionRuleIndices[j+1])
			{
				instSaver = csInstructionRules[j];
				indexsaver = csInstructionRuleIndices[j];
				csInstructionRules[j] = csInstructionRules[j+1];
				csInstructionRuleIndices[j] = csInstructionRuleIndices[j+1];
				csInstructionRules[j+1] = instSaver;
				csInstructionRuleIndices[j+1] = indexsaver;
			}
			else if(csInstructionRuleIndices[j] == csInstructionRuleIndices[j+1])
			{
				throw 50; //repeating indices
			}
		}
	}

	size = csDirectiveRulePrepList.size();
	csDirectiveRuleCount = size;
	csDirectiveRules = new DirectiveRule*[size];
	csDirectiveRuleIndices = new int[size];
	
	index = 0;
	for(list<DirectiveRule*>::iterator rule = csDirectiveRulePrepList.begin(); rule != csDirectiveRulePrepList.end(); rule++)
	{
		csDirectiveRules[index] = *rule;
		csDirectiveRuleIndices[index] = (*rule)->ComputeIndex();
		index++;
	}
	DirectiveRule *direSaver;

	//sort the indices
	for(int i = size - 1; i >  0; i--)
	{
		for(int j =0; j< i; j++)
		{
			//larger one behind
			if(csDirectiveRuleIndices[j] > csDirectiveRuleIndices[j+1])
			{
				direSaver = csDirectiveRules[j];
				indexsaver = csDirectiveRuleIndices[j];
				csDirectiveRules[j] = csDirectiveRules[j+1];
				csDirectiveRuleIndices[j] = csDirectiveRuleIndices[j+1];
				csDirectiveRules[j+1] = direSaver;
				csDirectiveRuleIndices[j+1] = indexsaver;
			}
			else if(csDirectiveRuleIndices[j] == csDirectiveRuleIndices[j+1])
			{
				throw 50; //repeating indices
			}
		}
	}



}
void Initialize() //set up the various lists
{
	//Set default parsers
	csMasterParserList.push_back	((MasterParser*)	&MPDefault);
	csLineParserList.push_back		((LineParser*)		&LPDefault);
	csInstructionParserList.push_back((InstructionParser*)&IPDefault);
	csDirectiveParserList.push_back	((DirectiveParser*)	&DPDefault);

	csMasterParser = (MasterParser*)&MPDefault;
	csLineParser = (LineParser*)	&LPDefault;
	csInstructionParser = (InstructionParser*)&IPDefault;
	csDirectiveParser = (DirectiveParser*)&DPDefault;
	
	//Load instruction rules
	//data movement
	csInstructionRulePrepList.push_back((InstructionRule*)&IRMOV);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRLD);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRLDU);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRLDL);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRLDC);;
	csInstructionRulePrepList.push_back((InstructionRule*)&IRLDS);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRST);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRSTL);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRSTS);
	//conversion
	csInstructionRulePrepList.push_back((InstructionRule*)&IRF2I);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRF2F);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRI2F);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRI2I);
	//execution control
	csInstructionRulePrepList.push_back((InstructionRule*)&IRBRA);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRCAL);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRPRET);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRRET);
	csInstructionRulePrepList.push_back((InstructionRule*)&IREXIT);
	//floating point op
	csInstructionRulePrepList.push_back((InstructionRule*)&IRFADD);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRFADD32I);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRFMUL);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRFFMA);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRFSETP);
	//integer opp
	csInstructionRulePrepList.push_back((InstructionRule*)&IRIADD);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRIADD32I);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRIMUL);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRIMAD);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRISETP);
	//Logic and shift
	csInstructionRulePrepList.push_back((InstructionRule*)&IRSHR);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRSHL);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRLOP);
	//miscellaneous
	csInstructionRulePrepList.push_back((InstructionRule*)&IRS2R);
	csInstructionRulePrepList.push_back((InstructionRule*)&IRNOP);

	//load directive rules
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRKernel);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DREndKernel);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRParam);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRShared);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRLocal);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRConstant2);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRConstant);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DREndConstant);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRSelfDebug);
	csDirectiveRulePrepList.push_back((DirectiveRule*)&DRArch);
	OrganiseRules();
}