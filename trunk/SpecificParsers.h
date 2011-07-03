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
	void Parse();
};
//-----End of default parser declarations


//	10
//-----Implementation of the parse() functions for parsers declared in 1.
void DefaultMasterParser:: Parse(unsigned int startinglinenumber)
{
	list<Line>::iterator cLine = csLines.begin(); //current line
	
	
	int lineLength;

	//Going through all lines
	for(unsigned int i =startinglinenumber; i<csLines.size(); i++)
	{
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
		cLine++;
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
	list<Component>::iterator component = csCurrentInstruction.Components.begin();
	if(component->Content[0]=='@')
	{
		csCurrentInstruction.Predicated = true;
		component++; processedComponent++;		
		if(component == csCurrentInstruction.Components.end())
			throw 100; //no instruction present while predicate is present
	}
	else
	{
		csCurrentInstruction.Predicated = false;
	}
	int nameIndex = hpParseComputeInstructionNameIndex(component->Content);
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

	if(csCurrentInstruction.Predicated)
		hpParseProcessPredicate();

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
		hpParseApplyModifier(*component, *csInstructionRules[arrayIndex]->ModifierRules[i]);
	}
	component++; processedComponent++;

	//here onwards are operands only. OPPresent is the number of operands that are present
	OPPresent = csCurrentInstruction.Components.size() - processedComponent;
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