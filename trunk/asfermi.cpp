//#include <vld.h> //Visual Leak Detector. You can just remove this when you compile.

#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <list>
#include "GlobalVariables.h"
#include "DataTypes.h"
#include "SpecificRules.h"
#include "helper.h"


using namespace std;

//-----Forward declarations
void ProcessCommandsAndReadFile(int argc, char** args);
void Initialize();
//extern void ExternInitialize();
void WriteToCubin();
//-----End of forward declarations



int main(int argc, char** args)
{
	try
	{
		//---Preprocess
		ProcessCommandsAndReadFile(argc, args);
		Initialize();
		//InitializeExtern();


		//---Process
		csMasterParser->Parse(0);


		//---Postprocess
		if(csErrorPresent)
			cout<<"Cannot proceed due to errors."<<endl;
		else
		{
			WriteToCubin();
			puts("Done");
		}
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

void WriteToCubin()
{
	//the following is for replace mode only. Independent generation of cubin is not done yet.
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

void ProcessCommandsAndReadFile(int argc, char** args)
{
	if(argc<6) //at least 6 strings: progpath inputpath -r target kernelname offset
	{
		csExceptionPrintUsage = true;
		throw 20; // invalid arguments
	}
	int currentArg = 1; //used to refer the next argument to be examined
	
	//Check input state. Single line instruction or file?
	if(strcmp(args[1], "-I") == 0) // Single line instruction. At least 7 args: progpath -I "instruction" -r target kernelname offset
	{
		if(argc<7)	//insufficient number of arguments
		{
			csExceptionPrintUsage = true;
			throw 20;
		}
		csSource = args[2]; //instruction string
		csLines.push_back( Line(SubString(0, strlen(csSource)), 0)); //directly create a single line
		currentArg = 3;
	}
	else //progpath inputpath -r target kernelname offset
	{
		hpReadSource(args[1]);	//Call helper function to read input file and create lines
		currentArg = 2;
	}
	//Input is read. Now proceed to output processing
	if(strcmp(args[currentArg], "-r")==0)	//replace mode
	{
		csReplaceMode = true;
		hpCheckOutput(args[currentArg+1], args[currentArg+2], args[currentArg+3]); //Use helper function to check validity of input arguments
						//csOutput is opened without closing 
	}
	else	//not replace mode
	{
		csReplaceMode = false;
		throw 99; //not in replacemode, not supported for now
	}

}

void OrganiseInstructionRules()
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
	InstructionRule *instsaver;
	int indexsaver;
	//sort the indices
	for(int i = size - 1; i >  0; i--)
	{
		for(int j =0; j< i; j++)
		{
			//larger one behind
			if(csInstructionRuleIndices[j] > csInstructionRuleIndices[j+1])
			{
				instsaver = csInstructionRules[j];
				indexsaver = csInstructionRuleIndices[j];
				csInstructionRules[j] = csInstructionRules[j+1];
				csInstructionRuleIndices[j] = csInstructionRuleIndices[j+1];
				csInstructionRules[j+1] = instsaver;
				csInstructionRuleIndices[j+1] = indexsaver;
			}
			else if(csInstructionRuleIndices[j] == csInstructionRuleIndices[j+1])
			{
				throw 50; //repeating indices
			}
		}
	}


}
void Initialize() //set up the various lists
{
	//Set default master parser
	DefaultMasterParser *dmp = new DefaultMasterParser();
	csMasterParserList.push_back(dmp);
	csMasterParser = dmp;
	
	//Set default line parser
	DefaultLineParser *dlp = new DefaultLineParser();
	csLineParserList.push_back(dlp);
	csLineParser = dlp;

	//Set default instruction parser
	DefaultInstructionParser *dip = new DefaultInstructionParser();
	csInstructionParserList.push_back(dip);
	csInstructionParser = dip;

	//Set default directive parser
	DefaultDirectiveParser *ddp = new DefaultDirectiveParser();
	csDirectiveParserList.push_back(ddp);
	csDirectiveParser = ddp;
	
	//Load instruction rules
	csInstructionRulePrepList.push_back(&IRLD);
	csInstructionRulePrepList.push_back(&IRST);
	csInstructionRulePrepList.push_back(&IREXIT);
	csInstructionRulePrepList.push_back(&IRMOV);
	csInstructionRulePrepList.push_back(&IRFADD);
	csInstructionRulePrepList.push_back(&IRIADD);
	::OrganiseInstructionRules();
}

//-----Implementation of the default parsers
void DefaultMasterParser:: Parse(unsigned int startinglinenumber)
{
	list<Line>::iterator cLine = csLines.begin(); //current line
	
	
	int lineLength;
	Instruction instruction;
	//Going through all lines
	for(unsigned int i =startinglinenumber; i<csLines.size(); i++)
	{
		lineLength = cLine->LineString.Length;
		//Jump to next line if there's nothing in this line
		if(lineLength==0)
			continue;
		if(cLine->LineString[0]=='!') //if it's directive, build it, parse it and append it to csDirectives
		{
			//build the new directive
			Directive directive(cLine->LineString, cLine->LineNumber);
			csDirectiveParser->Parse(directive);						//parse it. the parser will decide whether to append it to csDirectives or not
		}
		else //if it's not directive, it's instruction. Break it if it has ';'
		{
			try
			{
				//look for instruction delimiter ';'
				
				int startPos = 0;
				csCurrentInstruction = & instruction;//set this to avoid constant need to pass instruction around

				int lastfoundpos = cLine->LineString.Find(';', startPos); //search for ';', starting at startPos
				while(lastfoundpos!=-1)
				{
					//Build an instruction, parse it and the parser will decide whether to append it to csInstructions or not
					instruction.Reset(cLine->LineString.SubStr(startPos, lastfoundpos - startPos), csInstructionOffset, cLine->LineNumber);
					csInstructionParser->Parse();
					startPos = lastfoundpos + 1; //starting position of next search
					lastfoundpos = cLine->LineString.Find(';', startPos); //search for ';', starting at startPos
				}
				//still have to deal with the last part of the line, which may not end with ';'
				if(startPos < lineLength)
				{
					instruction.Reset(cLine->LineString.SubStr(startPos, lineLength - startPos), csInstructionOffset, cLine->LineNumber);
					csInstructionParser->Parse(); 
				}
			}
			catch(int e)
			{
				hpErrorHandler(e);
			}			
		}
		cLine++;
	}
}
//entire thing is done in the masterparser. So this is left empty for now.
void DefaultLineParser:: Parse(Line &line)
{
}


void DefaultInstructionParser:: Parse()
{
	hpBreakInstructionIntoComponents();

	
	//Start
	if(csCurrentInstruction->Components.size()==0)
		return;
	int processedComponent = 0;
	int OPPresent; //number of operands present
	list<Component>::iterator component = csCurrentInstruction->Components.begin();
	if(component->Content[0]=='@')
	{
		csCurrentInstruction->Predicated = true;
		component++; processedComponent++;		
		if(component == csCurrentInstruction->Components.end())
			throw 100; //no instruction present while predicate is present
	}
	else
	{
		csCurrentInstruction->Predicated = false;
	}
	int nameIndex = hpComputeInstructionNameIndex(component->Content);
	int arrayIndex = hpFindInstructionRuleArrayIndex(nameIndex);
	if(arrayIndex == -1)
	{
		throw 108; //instruction not supported
	}

	csCurrentInstruction->Is8 = csInstructionRules[arrayIndex]->Is8;
	csCurrentInstruction->OpcodeWord0 = csInstructionRules[arrayIndex]->OpcodeWord0;
	csInstructionOffset += 4;
	if(csCurrentInstruction->Is8)
	{
		csCurrentInstruction->OpcodeWord1 = csInstructionRules[arrayIndex]->OpcodeWord1;
		csInstructionOffset += 4;
	}

	if(csCurrentInstruction->Predicated)
		hpProcessPredicate();

	if(csInstructionRules[arrayIndex]->NeedCustomProcessing)
	{
		csInstructionRules[arrayIndex]->CustomProcess();
		goto APPEND;
	}

	for(list<SubString>::iterator modifier = component->Modifiers.begin(); modifier!=component->Modifiers.end(); modifier++)
	{
		int i;
		for(i = 0; i< csInstructionRules[arrayIndex]->ModifierCount; i++)
		{
			if(modifier->CompareWithCharArray(csInstructionRules[arrayIndex]->ModifierRules[i]->Name, csInstructionRules[arrayIndex]->ModifierRules[i]->Length))
				break;
		}
		if(i==csInstructionRules[arrayIndex]->ModifierCount)
		{
			throw 101; //unsupported modifier
			return;
		}
		hpApplyModifier(*component, *csInstructionRules[arrayIndex]->ModifierRules[i]);
	}
	component++; processedComponent++;

	//here onwards are operands only. OPPresent is the number of operands that are present
	OPPresent = csCurrentInstruction->Components.size() - processedComponent;
	if(OPPresent > csInstructionRules[arrayIndex]->OperandCount)
	{
		throw 102; //too many operands
		return;
	}
	
	for(int i=0; i<csInstructionRules[arrayIndex]->OperandCount; i++)
	{
		if(component == csCurrentInstruction->Components.end())
		{
			if(csInstructionRules[arrayIndex]->Operands[i]->Type == Optional)
				continue;
			else
			{
				throw 103; //insufficient operands.
			}
		}
		//process operand
		csInstructionRules[arrayIndex]->Operands[i]->Process(*component);
		component++;
		//process modifiers
		//not done yet
	}
APPEND:
	csInstructions.push_back(*csCurrentInstruction);
}

void DefaultDirectiveParser:: Parse(Directive &directive)
{
	csDirectives.push_back(directive);
}


//-----End of default parser implementations