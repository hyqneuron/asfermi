#if defined helperDefined //prevent multiple inclusion
#else
#define helperDefined yes
//---code starts ---
#include <vld.h>


#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <list>
#include "DataTypes.h"
#include "GlobalVariables.h"
#include "SpecificRules.h"




void hpUsage();

//	1
//-----Main helper functions
void hpExceptionHandler(int e)		//===
{
	char *message;
	switch(e)
	{
	case 0:		message = "Invalid arguments.";
		break;
	case 1:		message = "Unable to open input file";
		break;
	case 3:		message = "Incorrect kernel offset.";
		break;
	case 4:		message = "Cannot open output file";
		break;
	case 5:		message = "Cannot find specified kernel";
		break;
	case 6:		message = "File to be modified is invalid.";
		break;
	case 7:		message = "Failed to read cubin";
		break;
	case 8:		message = "Cannot find the specified section";
		break;
	case 9:		message = "Specific section not large enough to contain all the assembled opcodes.";
		break;
	case 20:	message = "Insufficient number of arguments";
		break;
	case 99:	message = "Not in replace mode.";
		break;
	default:	message = "No error message";
		break;
	};
	cout<<message<<endl;
	if(csExceptionPrintUsage)
		hpUsage();
}
void hpErrorHandler(int e, Instruction &instruction)
{
	csErrorPresent = true;
	char *message;
	switch(e)
	{
	case 100:	message = "Instruction name is absent following the predicate";
		break;
	case 101:	message = "Unsupported modifier.";
		break;
	case 102:	message = "Too many operands.";
		break;
	case 103:	message = "Insufficient number of operands.";
		break;
	case 104:	message = "Incorrect register format.";
		break;
	case 105:	message = "Register number too large.";
		break;
	case 106:	message = "Incorrect hex value.";
		break;
	case 107:	message = "Incorrect global memory format.";
		break;
	case 108:	message = "Instruction not supported.";
		break;
	default:	message = "Unknown Error";
		break;
	};
	char *line = instruction.InstructionString.ToCharArrayStopOnCR();
	cout<<"Line "<<instruction.LineNumber<<": "<<line<<": "<<message<<endl;
	delete[] line;
}

void hpCleanUp() //parsers are created with the new keyword,        =====
{
	for(list<MasterParser*>::iterator i = csMasterParserList.begin(); i!=csMasterParserList.end(); i++)
		delete *i;
	for(list<LineParser*>::iterator i = csLineParserList.begin(); i!=csLineParserList.end(); i++)
		delete *i;
	for(list<InstructionParser*>::iterator i = csInstructionParserList.begin(); i!=csInstructionParserList.end(); i++)
		delete *i;
	for(list<DirectiveParser*>::iterator i = csDirectiveParserList.begin(); i!=csDirectiveParserList.end(); i++)
		delete *i;

	delete[] csInstructionRules;
	delete[] csInstructionRuleIndices;

	delete[] csSource;
	if(csInput.is_open())
		csInput.close();
	if(csOutput.is_open())
		csOutput.close();
}
//-----End of main helper functions

//	2
//----- Command-line stage helper functions
void hpUsage()				//====
{
	puts("asfermi Version 0.0.1");
	puts("Usage:");
	puts("asfermi sourcefile [Options [option arguments]]");
	puts("Source file must be the first command-line option. However, it could be replaced by a -I option specified below.");
	puts("Options:");
	//puts("	-o outputfile: Output to the specified file. Not supported in this version.");
	puts("	-r target_cubin kernel_name offset: Replace the opcodes in specified location of a kernel in a specified cubin file with the assembled opcodes.");
	puts("	-I \"instruction\": This can be used to replace the inputfile. A single line of instruction, surrounded by double quotation marks, will be processed as source input.");
}



int hpHexCharToInt(char* str)			//====
{
	int result;
	//the string must start with "0x" or "0X"
	if(	str[0]!='0' ||
		!(str[1]=='x'|| str[1]=='X') ||
		(int)str[2]==0)
		throw 3; //incorrect kernel offset
	int numberlength = 0;
	stringstream ss;
	ss<<std::hex<<str;
	ss>>result;
	if(result==0) //when result = 0, str must be 0x00000..
	{
		for(int i =2; i<20; i++)
		{
			if(str[i]==0) //stop if end is reached.
				break;
			if(str[i]!='0') //when result is zero, all digitis ought to be '0' as well.
				throw 3;    //When this is not true, throws exception 3
		}
	}
	return result;
}


int hpFileSizeAndSetBegin(fstream &file)		//===
{
	file.seekg(0, fstream::end);
	int fileSize = file.tellg();
	file.seekg(0, fstream::beg);
	return fileSize;
}
void hpReadSource(char* path)				//===
{
	//Open file and check for validity of file
	csInput.open(path, fstream::in | fstream::binary);
	if(!csInput.is_open() | !csInput.good())
		throw 1; //Unable to open input file

	//Read all of file and close stream
	int fileSize = ::hpFileSizeAndSetBegin(csInput);
	csSource = new char[fileSize];
	csInput.read(csSource, fileSize);		//read all into csSource
	csInput.close();
	
	SubString entirety(0, fileSize);
	int startPos = 0;
	int length = 0;

	//Look for linefeed, (char)10. Length between startPos and lastLineFeedPos is returned in length
	int lastLineFeedPos = entirety.FindInSource(10, startPos, length);
	int lineNumber = 0;
	while(lastLineFeedPos!=-1)
	{
		csLines.push_back(Line(SubString(startPos, length), lineNumber++)); //Extract the found line and append to csLines
		startPos = lastLineFeedPos + 1;		//place startPos after the linefeed character
		lastLineFeedPos = entirety.FindInSource(10, startPos, length);
	}
	//deal with the last line
	if(startPos < fileSize)
		csLines.push_back(Line(SubString(startPos, fileSize - startPos), lineNumber++));
}

void hpCheckOutput(char* path, char* kernelname, char* replacepoint)		//===
{
	//open and check file
	csOutput.open(path, fstream::in | fstream::out | fstream::binary);
	if( !csOutput.is_open() || !csOutput.good() )
		throw 4; //can't open file

	//find file length and read entire file into buffer
	int fileSize = ::hpFileSizeAndSetBegin(csOutput);
	char* buffer = new char[fileSize];
	csOutput.read(buffer, fileSize);
	if(csOutput.bad() || csOutput.fail())
		throw 7; //failed to read cubin

	//Start checking the cubin and looking for sections
	if ( fileSize < 100 || (int)buffer[0] != 0x7f || (int)buffer[1] != 0x45 || (int)buffer[2] != 0x4c || (int)buffer[3] != 0x46) //ELF file identifier
		throw 6; //file to be modified is invalid
	//Reading the ELF header
	unsigned int SHTOffset =	*((unsigned int*)  (buffer+0x20));	//section header table offset, stored at 0x20, length 4
	unsigned int SHsize =		*((unsigned short*)(buffer+0x2e));	//section header size, stored at 0x2e, length 2
	unsigned int SHcount =		*((unsigned short*)(buffer+0x30));	//section header count
	unsigned int SHStrIdx =		*((unsigned short*)(buffer+0x32));	//section header string table index
	unsigned int SHStrOffset =	*((unsigned int*)  (buffer+SHTOffset+SHsize*SHStrIdx+0x10)); //offset in file of the section header string table, 0x10 is the offset in a header for section offset
	int fileoffset = SHTOffset;
	//Going through the section headers to look for the named section
	bool found = false;
	char *sectionname;
	for(unsigned int i =0; i<SHcount; i++, fileoffset+=SHsize)
	{
		unsigned int SHNameIdx =	*((unsigned int*)  (buffer+fileoffset)); //first 4-byte word in a section header is the name index
		sectionname = buffer + SHStrOffset + SHNameIdx;
		if(strcmp(sectionname, kernelname)==0)
		{
			found = true;
			csOutputSectionOffset = *((unsigned int*)  (buffer+fileoffset+0x10));
			csOutputSectionSize   = *((unsigned int*)  (buffer+fileoffset+0x14));
			csOutputInstructionOffset = hpHexCharToInt(replacepoint);
			delete[] buffer;
			return;
		}
	}
	delete[] buffer;
	throw 8;	//section not found
}
//-----End of command-line helper functions


//	3
//-----Parsing stage helper functions

int b_startPos;				//starting Position of a non-blank character in the instruction string
int b_currentPos;			//current search position in the instruction string. For the macros, this is the position of dot
int b_lineLength;			//length of the instruction string

//b_startPos = b_currentPos = position of first non-blank character found in [b_currentPos, b_lineLength)
//When no non-blank character is found, b_startPos is unmodified. b_currentPos = b_lineLength
#define mSkipBlank																							\
{																											\
	for(; b_currentPos < b_lineLength; b_currentPos++)														\
	{																										\
		if((int)instruction.InstructionString[b_currentPos]>32)												\
		{																									\
			b_startPos = b_currentPos;																		\
			break;																							\
		}																									\
	}																										\
	if(b_currentPos == b_lineLength)return;																									\
}

#define mExtract(startPos,cutPos) instruction.InstructionString.SubStr(startPos, cutPos-startPos)

void hpBreakInstructionIntoComponents(Instruction &instruction)
{
	b_currentPos = 0;
	b_startPos = 0;
	b_lineLength = instruction.InstructionString.Length;
	Component component;

	
//PREDSTART:
	mSkipBlank;
	if(instruction.InstructionString[b_currentPos]=='@')
	{
PRED:
		b_currentPos++;
		if(b_currentPos==b_lineLength)
		{
			component.Content = mExtract(b_startPos, b_currentPos);
			instruction.Components.push_back(component);
			return;
		}
		if(instruction.InstructionString[b_currentPos] < 33)
		{
			component.Content = mExtract(b_startPos, b_currentPos);
			instruction.Components.push_back(component);
			b_currentPos++; b_startPos = b_currentPos;
			mSkipBlank;
			goto INST;
		}
		else
			goto PRED;
	}

INST:
	if(b_currentPos==b_lineLength)
	{
		component.Content = mExtract(b_startPos, b_currentPos);
		instruction.Components.push_back(component);
		return;
	}
	else if( instruction.InstructionString[b_currentPos] < 33 )
	{
		component.Content = mExtract(b_startPos, b_currentPos);
		instruction.Components.push_back(component);
		b_currentPos++; b_startPos = b_currentPos;
		goto OP;
	}
	else if(instruction.InstructionString[b_currentPos] == '.')
	{
		component.Content = mExtract(b_startPos, b_currentPos);
		b_currentPos++; b_startPos = b_currentPos;
INSTDOT:
		if(b_currentPos==b_lineLength)
		{
			component.Modifiers.push_back( mExtract(b_startPos, b_currentPos));
			instruction.Components.push_back(component);
			return;
		}
		else if( instruction.InstructionString[b_currentPos] < 33 )
		{
			component.Modifiers.push_back( mExtract(b_startPos, b_currentPos));
			instruction.Components.push_back(component);
			b_currentPos++; b_startPos = b_currentPos;
		}
		else if(instruction.InstructionString[b_currentPos] == '.')
		{
			component.Modifiers.push_back( mExtract(b_startPos, b_currentPos));
			b_currentPos++; b_startPos = b_currentPos;
			goto INSTDOT;
		}
		else
		{
			b_currentPos++;
			goto INSTDOT;
		}
	}
	else
	{
		b_currentPos++;
		goto INST;
	}
OP:
	while(true)
	{
		mSkipBlank;
		component.Modifiers.clear();
OPREAL:
		if(b_currentPos==b_lineLength)
		{
			component.Content = mExtract(b_startPos, b_currentPos);
			instruction.Components.push_back(component);
			return;
		}
		else if( instruction.InstructionString[b_currentPos] == ',' )
		{
			component.Content = mExtract(b_startPos, b_currentPos);
			instruction.Components.push_back(component);
			b_currentPos++; b_startPos = b_currentPos;
			goto OP;
		}
		else if(instruction.InstructionString[b_currentPos] == '.')
		{
			component.Content = mExtract(b_startPos, b_currentPos);
			b_currentPos++; b_startPos = b_currentPos;
OPDOT:
			if(b_currentPos==b_lineLength)
			{
				component.Modifiers.push_back( mExtract(b_startPos, b_currentPos));
				instruction.Components.push_back(component);
				return;
			}
			else if( instruction.InstructionString[b_currentPos] == ',' )
			{
				component.Modifiers.push_back( mExtract(b_startPos, b_currentPos));
				instruction.Components.push_back(component);
				b_currentPos++; b_startPos = b_currentPos;
			}
			else if(instruction.InstructionString[b_currentPos] == '.')
			{
				component.Modifiers.push_back( mExtract(b_startPos, b_currentPos));
				b_currentPos++; b_startPos = b_currentPos;
				goto OPDOT;
			}
			else
			{
				b_currentPos++;
				goto OPDOT;
			}
		}
		else
		{
			b_currentPos++;
			goto OPREAL;
		}
	}

}


int hpComputeInstructionNameIndex(SubString &name)
{
	int len = name.Length;
	int index = 0;
	if(len>0)
	{
		index += (int)name[0] * 2851;
		if(len>1)
		{
			index += (int)name[1] * 349;
			for(int i =2; i<len; i++)
				index += (int)name[i];
		}
	}
	return index;
}
int hpFindInstructionRuleArrayIndex(int Index)
{
	int start = 0; //inclusive
	int end = csInstructionRuleCount; //exclusive
	int mid;
	while(start<end) //still got unchecked numbers
	{
		mid = (start+end)/2;
		if(Index > csInstructionRuleIndices[mid])
			start = mid + 1;
		else if(Index < csInstructionRuleIndices[mid])
			end = mid;
		else
			return mid;
	}
	return -1;
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
//-----End of parser helper functions

//9
//-----Debugging functions
void hpPrintLines()
{
	cout<<"Lines:"<<endl;
	for(list<Line>::iterator i = csLines.begin(); i!=csLines.end(); i++)
	{
		char * result = i->LineString.ToCharArray();
		cout<<result<<endl;
		delete[] result;
	}
}
void hpPrintInstructions()
{
	cout<<"Instructions"<<endl;
	for(list<Instruction>::iterator i = csInstructions.begin(); i!=csInstructions.end(); i++)
	{
		char * result = i->InstructionString.ToCharArray();
		cout<<result<<endl;
		delete[] result;
	}
}
void hpPrintDirectives()
{
	cout<<"Directives"<<endl;
	for(list<Directive>::iterator i = csDirectives.begin(); i!=csDirectives.end(); i++)
	{
		char * result = i->DirectiveString.ToCharArray();
		cout<<result<<endl;
		delete[] result;
	}
}

static const char* binaryRef[16] = {"0000", "1000", "0100", "1100", "0010", "1010", "0110", "1110", 
									"0001", "1001", "0101", "1101", "0011", "1011", "0111", "1111"};
void hpPrintBinary8(unsigned int word0, unsigned int word1)
{
	char *thisbyte = (char*) &word0;
	for(int i =0; i<4; i++)
	{
		unsigned char index = *(thisbyte++);
		cout<<binaryRef[index %16]<<binaryRef[index/16];
	}
	thisbyte = (char*) &word1;
	for(int i =0; i<4; i++)
	{
		unsigned char index = *(thisbyte++);
		cout<<binaryRef[index %16]<<binaryRef[index/16];
	}
	cout<<endl;
}
//-----End of debugging functions

#endif