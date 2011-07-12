/*
*/

//#include <vld.h> //Visual Leak Detector. You can just remove this when you compile.

#include <stdarg.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <list>

#include "SubString.h"
#include "DataTypes.h"
#include "Cubin.h"
#include "GlobalVariables.h"
#include "helper.h"
#include "helperParse.h"
#include "helperCubin.h"
#include "helperException.h"
#include "SpecificParsers.h"
#include "RulesModifier.h"
#include "RulesOperand.h"
#include "RulesInstruction.h"
#include "RulesDirective.h"
//#include "ExternRules.h" //this file can be used for adding custom rules


using namespace std;

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
	//tail sections: (not implemented).nv.constant2, .nv.info

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
	csMasterParserList.push_back(&DMP);
	csLineParserList.push_back(&DLP);
	csInstructionParserList.push_back(&DIP);
	csDirectiveParserList.push_back(&DDP);

	csMasterParser = &DMP;
	csLineParser = &DLP;
	csInstructionParser = &DIP;
	csDirectiveParser = &DDP;
	
	//Load instruction rules
	//data movement
	csInstructionRulePrepList.push_back(&IRMOV);
	csInstructionRulePrepList.push_back(&IRLD);
	csInstructionRulePrepList.push_back(&IRST);
	csInstructionRulePrepList.push_back(&IRLDS);
	csInstructionRulePrepList.push_back(&IRSTS);
	//execution control
	csInstructionRulePrepList.push_back(&IRBRA);
	csInstructionRulePrepList.push_back(&IRCAL);
	csInstructionRulePrepList.push_back(&IRPRET);
	csInstructionRulePrepList.push_back(&IRRET);
	csInstructionRulePrepList.push_back(&IREXIT);
	//floating point op
	csInstructionRulePrepList.push_back(&IRFADD);
	csInstructionRulePrepList.push_back(&IRFMUL);
	csInstructionRulePrepList.push_back(&IRFFMA);
	csInstructionRulePrepList.push_back(&IRFSETP);
	//integer opp
	csInstructionRulePrepList.push_back(&IRIADD);
	csInstructionRulePrepList.push_back(&IRIMUL);
	csInstructionRulePrepList.push_back(&IRIMAD);
	csInstructionRulePrepList.push_back(&IRISETP);
	//miscellaneous
	csInstructionRulePrepList.push_back(&IRS2R);
	csInstructionRulePrepList.push_back(&IRNOP);
	csInstructionRulePrepList.push_back(&IRLOP);

	//load directive rules
	csDirectiveRulePrepList.push_back(&DRKernel);
	csDirectiveRulePrepList.push_back(&DREndKernel);
	csDirectiveRulePrepList.push_back(&DRParam);
	csDirectiveRulePrepList.push_back(&DRSelfDebug);
	csDirectiveRulePrepList.push_back(&DRArch);
	OrganiseRules();
}