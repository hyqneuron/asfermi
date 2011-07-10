/*
This file contains the declaration of various parsers as well as the implementation of their parse() functions
1: Declaration of default parsers: DefaultMasterParser, DefaultLineParser, DefaultInstructionParser, DefaultDirectiveParser
10: Implementation of the parse() functions for parsers declared in 1.
*/

#ifndef SpecificParsersDefined //prevent multiple inclusion
//---code starts ---


using namespace std;


//	1
//-----Declaration of default parsers: DefaultMasterParser, DefaultLineParser, DefaultInstructionParser, DefaultDirectiveParser
struct DefaultMasterParser: MasterParser
{
	DefaultMasterParser()
	{
		Name = "DefaultMasterParser";
	}
	void Parse(unsigned int startinglinenumber);
}DMP;
struct DefaultLineParser : LineParser
{
	DefaultLineParser()
	{
		Name = "DefaultLineParser";
	}
	void Parse(Line &line);
}DLP;
struct DefaultInstructionParser: InstructionParser
{
	DefaultInstructionParser()
	{
		Name = "DefaultInstructionParser";
	}
	void Parse();
}DIP;
struct DefaultDirectiveParser: DirectiveParser
{
	DefaultDirectiveParser()
	{
		Name = "DefaultDirectiveParser";
	}
	void Parse();
}DDP;
//-----End of default parser declarations


//	10
//-----Implementation of the parse() functions for parsers declared in 1.
void DefaultMasterParser:: Parse(unsigned int startinglinenumber)
{
	list<Line>::iterator cLine = csLines.begin(); //current line	
	
	int lineLength;
	//Going through all lines
	for(unsigned int i =startinglinenumber; i<csLines.size(); i++, cLine++)
	{
		cLine->LineString.RemoveBlankAtBeginning();
		lineLength = cLine->LineString.Length;
		//Jump to next line if there's nothing in this line
		if(lineLength==0)
			continue;
		if(cLine->LineString[0]=='!') //if it's directive, build it, parse it and append it to csDirectives. Issue: the first character of the line must be '!'
		{
			try
			{
				//build the new directive
				csCurrentDirective.Reset(cLine->LineString, cLine->LineNumber);
				csDirectiveParser->Parse();						//parse it. the parser will decide whether to append it to csDirectives or not
			}
			catch(int e)
			{
				hpDirectiveErrorHandler(e);
			}
		}
		else //if it's not directive, it's instruction. Break it if it has ';'
		{
			try
			{
				//look for instruction delimiter ';'				
				int startPos = 0;
				int lastfoundpos = cLine->LineString.Find(';', startPos); //search for ';', starting at startPos
				while(lastfoundpos!=-1)
				{
					//Build an instruction, parse it and the parser will decide whether to append it to csInstructions or not
					csCurrentInstruction.Reset(cLine->LineString.SubStr(startPos, lastfoundpos - startPos), csInstructionOffset, cLine->LineNumber);
					csInstructionParser->Parse();
					startPos = lastfoundpos + 1; //starting position of next search
					lastfoundpos = cLine->LineString.Find(';', startPos); //search for ';', starting at startPos
				}
				//still have to deal with the last part of the line, which may not end with ';'
				if(startPos < lineLength)
				{
					csCurrentInstruction.Reset(cLine->LineString.SubStr(startPos, lineLength - startPos), csInstructionOffset, cLine->LineNumber);
					csInstructionParser->Parse();
				}
			}
			catch(int e)
			{
				hpInstructionErrorHandler(e);
			}			
		}
	}
}

//entire thing is done in the masterparser. So this is left empty for now.
void DefaultLineParser:: Parse(Line &line)
{
}


void DefaultInstructionParser:: Parse()
{
	hpParseBreakInstructionIntoComponents();

	
	//Start

	if(csCurrentInstruction.Components.size()==0)
		return;

	int processedComponent = 0;
	int OPPresent; //number of operands present
	list<SubString>::iterator component = csCurrentInstruction.Components.begin();
	list<SubString>::iterator modifier = csCurrentInstruction.Modifiers.begin();
	
	//---predicate
	if(csCurrentInstruction.Predicated)
	{
		hpParseProcessPredicate();
		component++; processedComponent++;		
	}

	//---instruction name
	if(component == csCurrentInstruction.Components.end())
			throw 100; //no instruction name present
	int nameIndex = hpParseComputeInstructionNameIndex(*component);
	int arrayIndex = hpParseFindInstructionRuleArrayIndex(nameIndex);
	if(arrayIndex == -1)
	{
		throw 108; //instruction not supported
	}

	csCurrentInstruction.Is8 = csInstructionRules[arrayIndex]->Is8;
	csCurrentInstruction.OpcodeWord0 = csInstructionRules[arrayIndex]->OpcodeWord0;
	csInstructionOffset += 4;
	if(csCurrentInstruction.Is8)
	{
		csCurrentInstruction.OpcodeWord1 = csInstructionRules[arrayIndex]->OpcodeWord1;
		csInstructionOffset += 4;
	}

	if(csInstructionRules[arrayIndex]->NeedCustomProcessing)
	{
		csInstructionRules[arrayIndex]->CustomProcess();
		goto APPEND;
	}

	




	//---instruction modifiers
	if(csCurrentInstruction.Modifiers.size()>csInstructionRules[arrayIndex]->ModifierGroupCount)
		throw 122;//too many modifiers.
	for(int modGroupIndex = 0; modGroupIndex < csInstructionRules[arrayIndex]->ModifierGroupCount; modGroupIndex++)
	{
		if(modifier==csCurrentInstruction.Modifiers.end())
		{
			//ignore if all the following groups are optional
			if(csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].Optional)
				continue;
			else
				throw 127; //insufficient number of modifiers
		}
		int i = 0;
		for( ; i < csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierCount; i++)
		{
			if(modifier->Compare(csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierRules[i]->Name))
				break;
		}
		//modifier name not found in this group
		if(i==csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierCount)
		{
			//Can ignore this modGroup if it is optional
			if(csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].Optional)
				continue;
			else
				throw 101; //unsupported modifier
		}
		hpParseApplyModifier(*csInstructionRules[arrayIndex]->ModifierGroups[modGroupIndex].ModifierRules[i]);
		modifier++;
	}	
	if(modifier!=csCurrentInstruction.Modifiers.end())
		throw 101; //issue: the error line should be something else
	component++; processedComponent++;



	//---Operands
	OPPresent = csCurrentInstruction.Components.size() - processedComponent; //OPPresent is the number of operands that are present
	if(OPPresent > csInstructionRules[arrayIndex]->OperandCount)
	{
		throw 102; //too many operands
		return;
	}
	
	for(int i=0; i<csInstructionRules[arrayIndex]->OperandCount; i++)
	{
		if(component == csCurrentInstruction.Components.end())
		{
			if(csInstructionRules[arrayIndex]->Operands[i]->Type == Optional)
				continue;
			else
				throw 103; //insufficient operands.
		}
		//process operand
		csInstructionRules[arrayIndex]->Operands[i]->Process(*component);
		component++;
		//process modifiers
		//not done yet
	}
APPEND:
	csInstructions.push_back(csCurrentInstruction);
}

void DefaultDirectiveParser:: Parse()
{
	hpParseBreakDirectiveIntoParts();
	if(csCurrentDirective.Parts.size()==0)
		throw 1000; //empty directive

	int index = hpParseComputeDirectiveNameIndex(*csCurrentDirective.Parts.begin());
	int arrayIndex = hpParseFindDirectiveRuleArrayIndex(index);
	if(arrayIndex==-1)
		throw 1001; //unsupported directive

	csDirectiveRules[arrayIndex]->Process();

	csDirectives.push_back(csCurrentDirective);
}

//-----End of parse() function implementation for default parsers
#else
#define SpecificParsersDefined
#endif