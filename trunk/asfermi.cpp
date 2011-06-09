 #include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "GlobalVariables.h"
#include "DataTypes.h"
#include "SpecificRules.h"
#include "helper.h"


using namespace std;

//-----Forward declarations
void ProcessCommandsAndReadFile(int argc, char** args);
void Initialize();
//void DefaultMasterParser::Parse(int startinglinenumber);
//void DefaultLineParser::Parse(Line line);
//void DefaultInstructionParser::Parse(Instruction instruction);
//void DefaultDirectiveParser::Parse(Directive directive);
//extern InitializeExtern();
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

		puts("=====");
		hpPrintLines();
		puts("=====");
		hpPrintInstructions();
		puts("=====");
		hpPrintDirectives();

		puts("======***======");
		list<Instruction>::iterator inst = csInstructions.begin();
		csOutput.seekp(csOutput.beg + ::csOutputSectionOffset + ::csOutputInstructionOffset);
		while(inst != csInstructions.end())
		{
			cout<<inst->Offset<<":";
			hpPrintBinary8(inst->OpcodeWord0, inst->OpcodeWord1);
			csOutput.write((char*)&inst->OpcodeWord0, 4);
			if(inst->Is8)
				csOutput.write((char*)&inst->OpcodeWord1, 4);
			inst++;
		}
		csOutput.flush();
		csOutput.close();
		puts("Done");


		hpCleanUp();
		getchar();
		return 0;
	}
	catch(int e)
	{
		hpExceptionHandler(e);
		return -1;
	}
}

void ProcessCommandsAndReadFile(int argc, char** args)
{
	//-----Start
	//Check argument length
	if(argc<6) //at least 6 strings: progpath inputpath -r target kernelname offset
	{
		throw 0; // invalid arguments
	}
	int currentArg = 1; //used to refer the next argument to be examined
	
	//Check input state. Single line instruction or file?
	if(strcmp(args[1], "-I") == 0) // Single line instruction. At least 7 args: progpath -I "instruction" -r target kernelname offset
	{
		if(argc<7)	//insufficient number of arguments
			throw 0;
		csSource = args[2]; //instruction string
		csLines.push_back( Line(SubString(0, strlen(csSource)), 0)); //directly create a single line
		currentArg = 3;
	}
	else //progpath inputpath -r target kernelname offset
	{
		::hpReadSource(args[1]);	//Call helper function to read input file and create lines
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
		throw 100; //not in replacemode, not supported for now
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
	::OrganiseInstructionRules();
}

//-----Implementation of the default parsers
void DefaultMasterParser:: Parse(unsigned int startinglinenumber)
{
	list<Line>::iterator cLine = csLines.begin(); //current line
	
	
	int lineLength;
	//Going through all lines
	for(int i =startinglinenumber; i<csLines.size(); i++)
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
			Instruction instruction;
			//look for instruction delimiter ';'
			
			int startPos = 0;
			int lastfoundpos = cLine->LineString.Find(';', startPos); //search for ';', starting at startPos
			while(lastfoundpos!=-1)
			{
				//Build an instruction, parse it and the parser will decide whether to append it to csInstructions or not
				Instruction instruction(cLine->LineString.SubStr(startPos, lastfoundpos - startPos), csInstructionOffset, cLine->LineNumber);
				csInstructionParser->Parse(instruction);

				startPos = lastfoundpos + 1; //starting position of next search
				lastfoundpos = cLine->LineString.Find(';', startPos); //search for ';', starting at startPos
			}
			//still have to deal with the last part of the line, which may not end with ';'
			if(startPos < lineLength)
			{
				instruction.Reset(cLine->LineString.SubStr(startPos, lineLength - startPos), csInstructionOffset, cLine->LineNumber);
				csInstructionParser->Parse(instruction); 
			}
		}
		cLine++;
	}
}
void DefaultLineParser:: Parse(Line &line)
{
}

void hpApplyModifier(Instruction &instruction, Component &component, ModifierRule &rule)
{
	if(rule.NeedCustomProcessing)
	{
		rule.CustomProcess(instruction, component);
	}
	else
	{
		if(rule.Apply0)
		{
			instruction.OpcodeWord0 &= rule.Mask0;
			instruction.OpcodeWord0 |= rule.Bits0;
		}
		if(instruction.Is8 && rule.Apply1)
		{
			instruction.OpcodeWord1 &= rule.Mask1;
			instruction.OpcodeWord1 |= rule.Bits1;
		}
	}
}
void DefaultInstructionParser:: Parse(Instruction &instruction)
{
	::hpPPtry(instruction);
	cout<<"--------"<<endl<<"Line "<<instruction.LineNumber<<": "<<instruction.InstructionString.ToCharArray()<<endl;
	for(list<Component>::iterator i = instruction.Components.begin(); i != instruction.Components.end(); i++)
	{
		cout<<i->Content.ToCharArray()<<"|| ";
		for(list<SubString>::iterator imod = i->Modifiers.begin(); imod != i->Modifiers.end(); imod++)
		{
			cout<< imod->ToCharArray()<<"|";
		}
		cout<<endl;
	}

	
	//csInstructions.push_back(instruction);
	
	//Start
	if(instruction.Components.size()==0)
		return;
	int processedComponent = 0;
	list<Component>::iterator component = instruction.Components.begin();
	if(component->Content[0]=='@')
	{
		instruction.Predicated = true;
		component++; processedComponent++;
	}
	else
	{
		instruction.Predicated = false;
	}
	if(component == instruction.Components.end())
	{
		throw 51; //no instruction present
		return;
	}
	int nameIndex = hpComputeInstructionNameIndex(component->Content);
	int arrayIndex = hpFindInstructionRuleArrayIndex(nameIndex);
	if(arrayIndex == -1)
	{
		throw exception(); //instruction not supported
		return;
	}
	if(csInstructionRules[arrayIndex]->NeedCustomProcessing)
	{
		csInstructionRules[arrayIndex]->CustomProcess(instruction);
		goto APPEND;
	}
	instruction.Is8 = csInstructionRules[arrayIndex]->Is8;
	instruction.OpcodeWord0 = csInstructionRules[arrayIndex]->OpcodeWord0;
	csInstructionOffset += 4;
	if(instruction.Is8)
	{
		instruction.OpcodeWord1 = csInstructionRules[arrayIndex]->OpcodeWord1;
		csInstructionOffset += 4;
	}

	for(list<SubString>::iterator modifier = component->Modifiers.begin(); modifier!=component->Modifiers.end(); modifier++)
	{
		int i;
		for(i = 0; i< csInstructionRules[arrayIndex]->ModifierCount; i++)
		{
			if(modifier->CompareWithCharArrayIgnoreEndingBlank(csInstructionRules[arrayIndex]->ModifierRules[i]->Name, csInstructionRules[arrayIndex]->ModifierRules[i]->Length))
				break;
		}
		if(i==csInstructionRules[arrayIndex]->ModifierCount)
		{
			throw 52; //modifier not found
			return;
		}
		hpApplyModifier(instruction, *component, *csInstructionRules[arrayIndex]->ModifierRules[i]);
	}
	component++; processedComponent++;

	//here onwards are operands only. OPPresent is the number of operands that are present
	int OPPresent = instruction.Components.size() - processedComponent;
	if(OPPresent > csInstructionRules[arrayIndex]->OperandCount)
	{
		throw 53; //too many operands
		return;
	}
	
	for(int i=0; i<csInstructionRules[arrayIndex]->OperandCount; i++)
	{
		if(component == instruction.Components.end())
		{
			if(csInstructionRules[arrayIndex]->Operands[i]->Type == Optional)
				continue;
			else
			{
				throw 54; //insufficient operands.
			}
		}
		//process operand
		csInstructionRules[arrayIndex]->Operands[i]->Process(instruction, *component);
		component++;
		//process modifiers
		//not done yet
	}
APPEND:
	csInstructions.push_back(instruction);
}

void DefaultDirectiveParser:: Parse(Directive &directive)
{
	csDirectives.push_back(directive);
}


//-----End of default parser implementations