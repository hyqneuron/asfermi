/*
 * Copyright (c) 2011, 2012 by Hou Yunqing and Dmitry Mikushin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <malloc.h>
#include <sstream>
#include <stack>

#include "SubString.h"
#include "DataTypes.h"
#include "Cubin.h"
#include "GlobalVariables.h"
#include "helper.h"
#include "SpecificParsers.h"

#include "RulesModifier.h"
#include "RulesOperand.h"
#include "RulesInstruction.h"
#include "RulesDirective.h"

#include "libasfermi.h"
#include "asfermi.h"

using namespace std;

void Initialize();
void WriteToCubinDirectOutput(iostream& csOutput);
void WriteRawOpcodes(iostream& csOutput);

static char* asfermi_encode(char* source, int cc, bool embed, int elf64bit, size_t* size)
{
	try
	{
		if (!source) return NULL;
		
		ASFermi asfermi;

		hpCubinSet64(elf64bit);

		Initialize();

		csSource = source;
		csOperationMode = DirectOutput;
	
		switch (cc)
		{
			case 20 : cubinArchitecture = sm_20; break;
			case 21 : cubinArchitecture = sm_21; break;
			case 30 : cubinArchitecture = sm_30; break;
			default : return NULL;
		}

		hpReadSourceArray(source);
		csMasterParser->Parse(0);
		if (csErrorPresent) throw 98;

		csSource = NULL;

		if (embed)
		{
			// Emit cubin to string stream and then copy
			// to plain char array.
			stringstream csOutput(stringstream::in | stringstream::out | stringstream::binary);
			WriteToCubinDirectOutput(csOutput);
			string str = csOutput.str();
			size_t length = str.size();
			char* result = (char*)malloc(length + 1);
			result[length] = '\0';
			memcpy(result, str.c_str(), length);
			if (size) *size = length;
			return result;
		}
		
		// Otherwise, emit opcodes.
		stringstream csOutput(stringstream::in | stringstream::out | stringstream::binary);
		WriteRawOpcodes(csOutput);
		string str = csOutput.str();
		size_t length = str.size();
		char* result = (char*)malloc(length + 1);
		result[length] = '\0';
		memcpy(result, str.c_str(), length);
		if (size) *size = length;
		return result;		
	}
	catch (int e)
	{
		hpExceptionHandler(e);
		hpCleanUp();
		return NULL;
	}

	hpCleanUp();
	
	fstream fstr;
	ifstream& ifstr1 = (ifstream&)fstr;
	stringstream sstr;
	ifstream& ifstr2 = (ifstream&)sstr;
}

char* asfermi_encode_cubin(char* source, int cc, int elf64bit, size_t* szcubin)
{
	return asfermi_encode(source, cc, true, elf64bit, szcubin);
}

char* asfermi_encode_opcodes(char* source, int cc, size_t* szopcodes)
{
	return asfermi_encode(source, cc, false, 0, szopcodes);
}

void WriteRawOpcodes(iostream& csOutput)
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
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		csOutput.write((char*)kernel->TextSection.SectionContent, kernel->TextSection.SectionSize);

		delete[] kernel->InfoSection.SectionContent;
		kernel->InfoSection.SectionContent = NULL;
		delete[] kernel->Constant0Section.SectionContent;
		kernel->Constant0Section.SectionContent = NULL;
		delete[] kernel->TextSection.SectionContent;
		kernel->TextSection.SectionContent = NULL;
	}
}

