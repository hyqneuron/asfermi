/*
 * Copyright (c) 2011, 2012 by Hou Yunqing
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

/*
This file contains various helper functions used during cubin output (the post-processin stage)
1: Stage1 functions
2: Stage2 functions
3: Stage3 functions
4: Stage4 and later functions

all functions are prefixed with 'hpCubin'
*/
#include "helperCubin.h"
#include "../GlobalVariables.h"
#include "../Cubin.h"
#include "../DataTypes.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

void hpCubinSet64(bool set64)
{
	if(set64)
	{
		ELFHeaderSize = 0x40;
		ELFSectionHeaderSize=0x40;
		ELFSegmentHeaderSize=0x38;
		ELFSymbolEntrySize=0x18;
		cubin64Bit = true;
	}
	else
	{
		ELFHeaderSize = 0x34;
		ELFSectionHeaderSize=0x28;
		ELFSegmentHeaderSize=0x20;
		ELFSymbolEntrySize=0x10;
		cubin64Bit = false;
	}
}


//	1
//-----Stage1 functions
//SectionIndex, SectionSize, OffsetFromFirst, SHStrTabOffset 
void hpCubinStage1SetSection(ELFSection &section, SectionType nvType, unsigned int kernelNameLength)
{
	//Index
	section.SectionIndex = cubinCurrentSectionIndex++;
	
	//SHStrTabOffset
	section.SHStrTabOffset = cubinCurrentSHStrTabOffset;
	cubinCurrentSHStrTabOffset += kernelNameLength;
	switch(nvType)
	{
	case KernelText:
		cubinCurrentSHStrTabOffset += 7; //".text." + 0
		break;
	case KernelConstant0:
		cubinCurrentSHStrTabOffset += 15; //".nv.constant0." + 0
		break;
	case KernelInfo:
		cubinCurrentSHStrTabOffset += 10; //".nv.info."  + 0
		break;
	case KernelShared:
		cubinCurrentSHStrTabOffset += 12; //".nv.shared." + 0
		break;
	case KernelLocal:
		cubinCurrentSHStrTabOffset += 11; //".nv.local." + 0
		break;
	default:
		throw 9999; //issue: not included yet
	};
}
void hpCubinStage1()
{
	//Setup SectionIndex, SHStrTabOffset for .shstrtab, .strtab, .symtab
	cubinCurrentSHStrTabOffset = 1; //jump over first null character
	cubinCurrentSectionIndex = 1; //jump over first null section

	cubinSectionSHStrTab.SectionIndex = cubinCurrentSectionIndex++;
	cubinSectionSHStrTab.SHStrTabOffset = cubinCurrentSHStrTabOffset;
	cubinCurrentSHStrTabOffset += strlen(cubin_str_shstrtab) + 1; //increment by length of name + length of the ending zero

	
	cubinSectionStrTab.SectionIndex = cubinCurrentSectionIndex++;
	cubinSectionStrTab.SHStrTabOffset = cubinCurrentSHStrTabOffset;
	cubinCurrentSHStrTabOffset += strlen(cubin_str_strtab) + 1;

	
	cubinSectionSymTab.SectionIndex = cubinCurrentSectionIndex++;
	cubinSectionSymTab.SHStrTabOffset = cubinCurrentSHStrTabOffset;
	cubinCurrentSHStrTabOffset += strlen(cubin_str_symtab) + 1;


	//Setup SectionIndex, SHStrTabOffset for all sections of all kernels
	//Setup StrTabOffset for all kernels
	
	cubinCurrentStrTabOffset = 1; //jump over first null character
	for(list<Kernel>::iterator kernel = csKernelList.begin(),
		kernele = csKernelList.end(); kernel != kernele; kernel++)
	{
		//Text
		hpCubinStage1SetSection(kernel->TextSection, KernelText, kernel->KernelName.Length);
		//Constant0
		hpCubinStage1SetSection(kernel->Constant0Section, KernelConstant0, kernel->KernelName.Length);
		//Info
		int infoSize = 12;
		if(kernel->Parameters.size()!=0)
			infoSize = 0x14 * (kernel->Parameters.size()+1);
		hpCubinStage1SetSection(kernel->InfoSection, KernelInfo, kernel->KernelName.Length);
		//Shared
		if(kernel->SharedSize!=0)
			hpCubinStage1SetSection(kernel->SharedSection, KernelShared, kernel->KernelName.Length);
		//Local
		if(kernel->LocalSize!=0)
			hpCubinStage1SetSection(kernel->LocalSection, KernelLocal, kernel->KernelName.Length);
		//StrTabOffset
		kernel->StrTabOffset = cubinCurrentStrTabOffset;

		//increment by length of kernel name + length of endng zero
		cubinCurrentStrTabOffset += kernel->KernelName.Length + 1;
	}
	for (list<Constant2>::iterator constant2 = csConstant2List.begin(),
		constant2e = csConstant2List.end(); constant2 != constant2e; constant2++)
	{
		//StrTabOffset
		constant2->StrTabOffset = cubinCurrentStrTabOffset;
		//increment by length of constant name + length of endng zero
		cubinCurrentStrTabOffset += constant2->Constant2Name.size() + 1;
	}
	//Setup SectionIndex, SHStrTabOffset for .nv.info, nv.constant2
	
	if(cubinConstant2Size)
	{
		cubinSectionConstant2.SectionIndex = cubinCurrentSectionIndex++;
		cubinSectionConstant2.SHStrTabOffset = cubinCurrentSHStrTabOffset;
		cubinCurrentSHStrTabOffset += strlen(cubin_str_constant) + 2;
	}

	cubinSectionNVInfo.SectionIndex = cubinCurrentSectionIndex++;
	cubinSectionNVInfo.SHStrTabOffset = cubinCurrentSHStrTabOffset;
	cubinCurrentSHStrTabOffset += strlen(cubin_str_nvinfo) + 1;
	//cubinCurrentSHStrTabOffset:	size of shstrtab
	//cubinCurrentStrTabOffset:		size of strtab
	//cubinCurrentSectionIndex:		total section count
}














//	2
//-----Stage2 functions


inline void hpCubinAddSectionName1(unsigned char* sectionContent, int &offset, char* sectionPrefix, SubString &kernelName)
{
	memcpy(sectionContent + offset, sectionPrefix, strlen(sectionPrefix));
	offset += strlen(sectionPrefix);
	memcpy(sectionContent + offset, kernelName.Start, kernelName.Length);
	offset += kernelName.Length;
	*(char*)(sectionContent + offset) = (char)0;
	offset += 1;
}
inline void hpCubinAddSectionName2(unsigned char* sectionContent, int &offset, const char* sectionName)
{
	memcpy(sectionContent + offset, sectionName, strlen(sectionName) + 1);
	offset += strlen(sectionName) + 1;
}
inline void hpCubinAddSectionName3(unsigned char* sectionContent, int &offset, SubString &kernelName)
{
	memcpy(sectionContent + offset, kernelName.Start, kernelName.Length);
	offset += kernelName.Length;
	*(char*)(sectionContent + offset) = (char)0;
	offset += 1;
}
inline void hpCubinStage2SetSHStrTabSectionContent()
{
	cubinSectionSHStrTab.SectionSize = cubinCurrentSHStrTabOffset;
	cubinSectionSHStrTab.SectionContent = new unsigned char[cubinCurrentSHStrTabOffset];

	cubinSectionSHStrTab.SectionContent[0] = 0;
	int currentOffset = 1;
	//head sections
	hpCubinAddSectionName2(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_shstrtab);
	hpCubinAddSectionName2(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_strtab);
	hpCubinAddSectionName2(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_symtab);
	//kern sections
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		hpCubinAddSectionName1(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_text, kernel->KernelName);
		hpCubinAddSectionName1(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_constant0, kernel->KernelName);
		hpCubinAddSectionName1(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_info, kernel->KernelName);
		if(kernel->SharedSize)
			hpCubinAddSectionName1(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_shared, kernel->KernelName);
		if(kernel->LocalSize)
			hpCubinAddSectionName1(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_local, kernel->KernelName);
	}
	//tail sections
	if(cubinConstant2Size)
	{
		// Switch between .nv.constant2 and .nv.constant3, depending on target architecture.
		string name = cubin_str_constant;
		if(cubinArchitecture == sm_20||cubinArchitecture ==sm_21)
			name += "2";
		else if(cubinArchitecture ==sm_30)
			name += "3";
		hpCubinAddSectionName2(cubinSectionSHStrTab.SectionContent, currentOffset, name.c_str());
	}
	hpCubinAddSectionName2(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_nvinfo);
}
inline void hpCubinStage2SetStrTabSectionContent()
{
	cubinSectionStrTab.SectionSize = cubinCurrentStrTabOffset;
	cubinSectionStrTab.SectionContent = new unsigned char[cubinCurrentStrTabOffset];

	cubinSectionStrTab.SectionContent[0] = 0;
	int currentOffset = 1;

	// 1 entry for each kernel
	for(list<Kernel>::iterator kernel = csKernelList.begin(),
		kernele = csKernelList.end(); kernel != kernele; kernel++)
		hpCubinAddSectionName3(cubinSectionStrTab.SectionContent,
			currentOffset, kernel->KernelName);

	// 1 entry for each constant
	for (list<Constant2>::iterator constant2 = csConstant2List.begin(),
		constant2e = csConstant2List.end(); constant2 != constant2e; constant2++)
		hpCubinAddSectionName2(cubinSectionStrTab.SectionContent,
			currentOffset, constant2->Constant2Name.c_str());
}

inline void hpCubinStage2AddSectionSymbol(ELFSection &section, ELFSymbolEntry &entry, int &index, unsigned int size)
{
		entry.Reset();
		entry.Size = size;
		entry.Info = 3;
		entry.SHIndex = section.SectionIndex;
		((ELFSymbolEntry*)cubinSectionSymTab.SectionContent)[index] = entry;
		section.SymbolIndex = index++;
}
inline void hpCubinStage2SetSymTabSectionContent()
{
	// 1 entry for each section, 1 entry for each kernel,
	// 1 entry for each constant, 2 empty entries.		
	int entryCount = cubinCurrentSectionIndex + csKernelList.size() +
		+ csConstant2List.size() + 2;
	cubinSectionSymTab.SectionSize = entryCount * ELFSymbolEntrySize;
	ELFSymbolEntry* entries = new ELFSymbolEntry[cubinSectionSymTab.SectionSize];
	cubinSectionSymTab.SectionContent = (unsigned char*)entries;

	//first 6 entries
	memset(cubinSectionSymTab.SectionContent, 0, cubinSectionSymTab.SectionSize); //clear everything to 0 first
	//jump over the entry 0 (null), to directly to entry 1
	//set symbol for head sections
	entries[1].SHIndex = 1; //only setting section index and info, leaving other things zero
	entries[2].SHIndex = 2;
	entries[3].SHIndex = 3;

	entries[1].Info = 3;
	entries[2].Info = 3;
	entries[3].Info = 3;
	entries[4].Info = 3;
	entries[5].Info = 3;

	//one entry per kern section	
	int index = 6; //jump over entry 4 and 5 which are empty
	ELFSymbolEntry entry;
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		//text
		hpCubinStage2AddSectionSymbol(kernel->TextSection, entry, index, kernel->TextSize);
		//constant0
		hpCubinStage2AddSectionSymbol(kernel->Constant0Section, entry, index, 0);
		//info
		hpCubinStage2AddSectionSymbol(kernel->InfoSection, entry, index, 0);
		//shared section
		if(kernel->SharedSize>0)
			hpCubinStage2AddSectionSymbol(kernel->SharedSection, entry, index, 0);
		//local section
		if(kernel->LocalSize>0)
			hpCubinStage2AddSectionSymbol(kernel->LocalSection, entry, index, 0);
	}
	//tail sections
	if(cubinConstant2Size)
		hpCubinStage2AddSectionSymbol(cubinSectionConstant2, entry, index, 0);
	hpCubinStage2AddSectionSymbol(cubinSectionNVInfo, entry, index, 0);

	//one entry per __global__ function
	for(list<Kernel>::iterator kernel = csKernelList.begin(),
		kernele = csKernelList.end(); kernel != kernele; kernel++)
	{
		entry.Name = kernel->StrTabOffset;
		entry.Value = 0;
		entry.Size = kernel->TextSize;
		entry.Info = 0x12;
		entry.Other = 0x10;
		entry.SHIndex = kernel->TextSection.SectionIndex;
		entries[index] = entry;
		kernel->GlobalSymbolIndex = index++;
	}

	//one entry per constant symbol
        for (list<Constant2>::iterator constant2 = csConstant2List.begin(),
                constant2e = csConstant2List.end(); constant2 != constant2e; constant2++)
	{
		entry.Name = constant2->StrTabOffset;

		// Since constant data layout is defined by offsets (not sizes),
		// here we need to determine the size of the constant symbol:
		// the distance between the current symbol offset and the minimum
		// offset that is greater than current (or section size)
		entry.Size = cubinSectionConstant2.SectionSize - constant2->Offset;
		for (list<Constant2>::iterator constant2other = csConstant2List.begin();
			constant2other != constant2e; constant2other++)
		{
			if (constant2e == constant2other) continue;
		
			int size = constant2other->Offset - constant2->Offset;
			if ((size > 0) && (size < entry.Size)) entry.Size = size;
		}

		entry.Value = constant2->Offset;
		entry.Info = 0x11;
		entry.Other = 0x0;
		entry.SHIndex = cubinSectionConstant2.SectionIndex;
		entries[index++] = entry;
	}
}
void hpCubinStage2()
{
	//---.shstrtab
	hpCubinStage2SetSHStrTabSectionContent();

	//---.strtab	
	hpCubinStage2SetStrTabSectionContent();

	//---.symtab
	hpCubinStage2SetSymTabSectionContent();
}








//	3
//-----Stage3
void hpCubinStage3()
{
	//kern sections
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		//.text
		kernel->TextSection.SectionSize = kernel->TextSize; //device functions not implemented. When implemented, this has to include size of device func as well
		kernel->TextSection.SectionContent = new unsigned char[kernel->TextSection.SectionSize];
		unsigned int *offset = (unsigned int *)(kernel->TextSection.SectionContent);
		for(list<Instruction>::iterator inst = kernel->KernelInstructions.begin(); inst != kernel->KernelInstructions.end(); inst++)
		{
			*offset = inst->OpcodeWord0;
			offset ++;
			if(inst->Is8)
			{
				*offset = inst->OpcodeWord1;
				offset ++;
			}
		}
		int didsize = (unsigned char*)offset - kernel->TextSection.SectionContent;
		if( (didsize) != kernel->TextSection.SectionSize)
		{
			throw;
		}

		//.constant0
		if(cubinArchitecture == sm_20||cubinArchitecture ==sm_21) {
			kernel->Constant0Section.SectionSize = 0x20 + kernel->ParamTotalSize;
			kernel->Constant0Section.SectionContent = new unsigned char[kernel->Constant0Section.SectionSize];
			memset(kernel->Constant0Section.SectionContent, 0, kernel->Constant0Section.SectionSize); //just set it all to 0
		}
		else if(cubinArchitecture ==sm_30) {
			kernel->Constant0Section.SectionSize = 0x140 + kernel->ParamTotalSize;
			kernel->Constant0Section.SectionContent = new unsigned char[kernel->Constant0Section.SectionSize];
			memset(kernel->Constant0Section.SectionContent, 0, kernel->Constant0Section.SectionSize); //just set it all to 0
		}

		//.info
		if(kernel->Parameters.size()==0) //no param
		{
			kernel->InfoSection.SectionSize = 12;
			kernel->InfoSection.SectionContent = new unsigned char[12];
			offset = (unsigned int *)kernel->InfoSection.SectionContent;
			//param_cbank
			*offset++ = 0x00080a04; //identifier: 04 0a 08 00
			*offset++ = kernel->Constant0Section.SymbolIndex; //next value is constant0 section symbol index
			*offset = 0x200000;	//without param, this is always 0x00200000
		}
		else
		{
			if(cubinArchitecture ==sm_21) {
				kernel->InfoSection.SectionSize = 0x14 * (kernel->Parameters.size() + 1);//size = (n+1)(0x14)
			}
			else if(cubinArchitecture == sm_20||cubinArchitecture == sm_30) {
				kernel->InfoSection.SectionSize = 0x10 * (kernel->Parameters.size() + 1);//size = (n+1)(0x14)
			}
			kernel->InfoSection.SectionContent = new unsigned char[kernel->InfoSection.SectionSize];
			offset = (unsigned int *)kernel->InfoSection.SectionContent;

			if(cubinArchitecture ==sm_21) {
				//---cbank_param_offsets
				*offset++ = 0x00000c04 | kernel->Parameters.size()*4 << 16; //04 0c aa bb: bbaa is paramcount * 4
				//offset of each argument
				for(list<KernelParameter>::iterator param = kernel->Parameters.begin();
					param != kernel->Parameters.end(); param++)
					*offset++ = param->Offset;
			}

			//---param_cbank
			*offset++ = 0x00080a04; //size to follow is always 08
			*offset++ = kernel->Constant0Section.SymbolIndex;
			if(cubinArchitecture == sm_20||cubinArchitecture ==sm_21) {
				*offset++ = kernel->ParamTotalSize << 16 | 0x0020; //0x00aa0020: 0xaaaa: total parameter size
			}
			else if(cubinArchitecture ==sm_30) {
				*offset++ = kernel->ParamTotalSize << 16 | 0x0140; //0x00aa0140: 0xaaaa: total parameter size
			}

			//---cbank_param_size
			*offset++ = 0x00001903 | kernel->ParamTotalSize << 16; //03 19 aa bb: 0xbbaa: total param size

			//---kparam_info
			unsigned int ordinal = kernel->Parameters.size() - 1; //starts from the end of the param list
			for(list<KernelParameter>::reverse_iterator param = kernel->Parameters.rbegin(); param != kernel->Parameters.rend(); param++)
			{
				*offset++ = 0x000c1704; //identifier: 04 17 0c 00
				*offset++ = 0x0; //index, always -0x1
				//*offset++ = 0xffffffff; //index, always -0x1
				*offset++ = ( ordinal-- )| (param->Offset<<16); // aa bb cc dd: bbaa is ordinal, ddcc is offset
				*offset++ = (((param->Size+3)/4)<<20)|0x0001f000; //aaa b c b dd: aaa is size of param/4, bb is cbank, c is space, dd is logAlignment
			}
		}

		//.shared
		kernel->SharedSection.SectionSize = kernel->SharedSize;
		//.local
		kernel->LocalSection.SectionSize = kernel->LocalSize;
	}
	

	//tail sections
	//.nv.constant2, sectionContent is set up by directive
	if(cubinConstant2Size)
		cubinSectionConstant2.SectionSize = cubinConstant2Size;
	
	//.nv.info
	cubinSectionNVInfo.SectionSize = 0x18 * csKernelList.size();//it's guaranteed by the caller that size is greater than 0
	cubinSectionNVInfo.SectionContent = new unsigned char[cubinSectionNVInfo.SectionSize]; 
	unsigned int *offset = (unsigned int *)cubinSectionNVInfo.SectionContent;
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		*offset++ = 0x00081204; //identifier: 04 12 08 00
		*offset++ = kernel->GlobalSymbolIndex;
		*offset++ = kernel->MinStackSize;
		
		*offset++ = 0x00081104; // 04 11 08 00
		*offset++ = kernel->GlobalSymbolIndex;
		*offset++ = kernel->MinFrameSize;
	}
}



















//	4
//-----Stage4 and later functions

void hpCubinSetELFSectionHeader1(ELFSection &section, unsigned int type, unsigned int alignment, unsigned int &offset, bool moveOffset)
{
	memset(&section.SectionHeader, 0, sizeof(ELFSectionHeader));
	section.SectionHeader.NameIndex = section.SHStrTabOffset;
	section.SectionHeader.Type = type;
	section.SectionHeader.FileOffset = offset;
	section.SectionHeader.Size = section.SectionSize;
	section.SectionHeader.Alignment = alignment;
	if(moveOffset)
		offset += section.SectionSize;
}
void hpCubinSetELFSectionHeader1(ELFSection &section, unsigned int type, unsigned int alignment, unsigned int &offset)
{
	hpCubinSetELFSectionHeader1(section, type, alignment,offset, true);
}

//Stage4: Setup all section headers
void hpCubinStage4()
{
	//unsigned int fileOffset = (cubin64Bit?0x40:0x34) + 0x28 * cubinCurrentSectionIndex; //start of the shstrtab section content
	unsigned int fileOffset = ELFHeaderSize + ELFSectionHeaderSize * cubinCurrentSectionIndex; //start of the shstrtab section content

	//---head sections
	//empty
	memset(&cubinSectionEmpty.SectionHeader, 0, sizeof(ELFSectionHeader));
	//shstrtab
	hpCubinSetELFSectionHeader1(cubinSectionSHStrTab, 3, 4, fileOffset);
	//strtab
	hpCubinSetELFSectionHeader1(cubinSectionStrTab, 3, 1, fileOffset);
	//symtab
	hpCubinSetELFSectionHeader1(cubinSectionSymTab, 2, 1, fileOffset);
	cubinSectionSymTab.SectionHeader.EntrySize = ELFSymbolEntrySize;
	cubinSectionSymTab.SectionHeader.Info = cubinCurrentSectionIndex+2; //info is number of local symbols
	cubinSectionSymTab.SectionHeader.Link = 2;

	//---kern sections
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		//.text
		hpCubinSetELFSectionHeader1(kernel->TextSection, 1, 4, fileOffset);
		kernel->TextSection.SectionHeader.Flags = 6 | kernel->BarCount<<20; //20:26 are bar count
		kernel->TextSection.SectionHeader.Link = 3;
		kernel->TextSection.SectionHeader.Info = kernel->TextSection.SymbolIndex | kernel->RegCount << 24; //highest byte is number of reg
		//.constant0
		hpCubinSetELFSectionHeader1(kernel->Constant0Section, 1, 4, fileOffset);
		kernel->Constant0Section.SectionHeader.Flags = 2;
		kernel->Constant0Section.SectionHeader.Info = kernel->TextSection.SectionIndex;
		//.info
		hpCubinSetELFSectionHeader1(kernel->InfoSection, 1, 1, fileOffset);
		kernel->InfoSection.SectionHeader.Flags = 2;
		kernel->InfoSection.SectionHeader.Info = kernel->TextSection.SectionIndex;
		//.shared
		if(kernel->SharedSize>0)
		{
			hpCubinSetELFSectionHeader1(kernel->SharedSection, 8, 4, fileOffset, false);
			kernel->SharedSection.SectionHeader.Flags = 3;
			kernel->SharedSection.SectionHeader.Info = kernel->TextSection.SectionIndex;
		}
		//.local
		if(kernel->LocalSize>0)
		{
			hpCubinSetELFSectionHeader1(kernel->LocalSection, 8, 4, fileOffset, false);
			kernel->LocalSection.SectionHeader.Flags = 3;
			kernel->LocalSection.SectionHeader.Info = kernel->TextSection.SectionIndex;
		}
	}
	//---tail sections
	//nv.constant2
	if(cubinConstant2Size)
		hpCubinSetELFSectionHeader1(cubinSectionConstant2, 1, 4, fileOffset);
	//.nv.info
	hpCubinSetELFSectionHeader1(cubinSectionNVInfo, 1, 1, fileOffset);
	cubinSectionNVInfo.SectionHeader.Flags = 2;

	cubinPHTOffset = fileOffset;
}

//Stage5: Setup all program segments
void hpCubinStage5()
{
	int count = 0;
	//kern segments
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		kernel->KernelSegmentHeader.Type = 0x60000000;
		kernel->KernelSegmentHeader.Offset = kernel->TextSection.SectionHeader.FileOffset;
		kernel->KernelSegmentHeader.FileSize = kernel->TextSize + kernel->Constant0Section.SectionSize + kernel->InfoSection.SectionSize;
		kernel->KernelSegmentHeader.MemSize = kernel->KernelSegmentHeader.FileSize;
		kernel->KernelSegmentHeader.Flags = 0x05 | kernel->GlobalSymbolIndex<<8;
		kernel->KernelSegmentHeader.Alignment = 4;
		kernel->KernelSegmentHeader.PhysicalMemAddr = 0;
		kernel->KernelSegmentHeader.VirtualMemAddr = 0;
		count ++;

		if(kernel->SharedSize != 0 || kernel->LocalSize !=0)
		{
			kernel->MemorySegmentHeader.Type = 0x60000000;
			if(kernel->SharedSize != 0)
				kernel->MemorySegmentHeader.Offset = kernel->SharedSection.SectionHeader.FileOffset;
			else
				kernel->MemorySegmentHeader.Offset = kernel->LocalSection.SectionHeader.FileOffset;
			kernel->MemorySegmentHeader.FileSize = 0;
			kernel->MemorySegmentHeader.MemSize = kernel->SharedSize + kernel->LocalSize;
			kernel->MemorySegmentHeader.Flags = 0x06 | kernel->GlobalSymbolIndex<<8;
			kernel->MemorySegmentHeader.Alignment = 4;
			kernel->MemorySegmentHeader.PhysicalMemAddr = 0;
			kernel->MemorySegmentHeader.VirtualMemAddr = 0;
			count ++;
		}
	}

	//.nv.constant2
	if(cubinConstant2Size)
	{
		cubinSegmentHeaderConstant2.Type = 1;
		cubinSegmentHeaderConstant2.Offset = cubinSectionConstant2.SectionHeader.FileOffset;
		cubinSegmentHeaderConstant2.FileSize = cubinConstant2Size;
		cubinSegmentHeaderConstant2.MemSize = cubinConstant2Size;
		cubinSegmentHeaderConstant2.Flags = 5;
		cubinSegmentHeaderConstant2.Alignment = 4;
		cubinSegmentHeaderConstant2.VirtualMemAddr = 0;
		cubinSegmentHeaderConstant2.PhysicalMemAddr = 0;
		count++;
	}
	
	cubinPHCount = count + 1; // +1 is the SELF
	//PHTSelf
	cubinSegmentHeaderPHTSelf.Type = 6;
	cubinSegmentHeaderPHTSelf.Flags = 5;
	cubinSegmentHeaderPHTSelf.Alignment = 4;
	cubinSegmentHeaderPHTSelf.Offset = cubinPHTOffset;
	cubinSegmentHeaderPHTSelf.FileSize = ELFSegmentHeaderSize * cubinPHCount;
	cubinSegmentHeaderPHTSelf.MemSize = cubinSegmentHeaderPHTSelf.FileSize;
	cubinSegmentHeaderPHTSelf.VirtualMemAddr = 0;
	cubinSegmentHeaderPHTSelf.PhysicalMemAddr = 0;
}

//Stage6: Setup ELF header
void hpCubinStage6()
{
	//issue: supports only sm_20, sm_21 & sm_30
	if(cubinArchitecture == sm_20)
		ELFH32.Flags = ELFFlagsForsm_20;
	else if (cubinArchitecture == sm_21) 
		ELFH32.Flags = ELFFlagsForsm_21;
	else 
		ELFH32.Flags = ELFFlagsForsm_30;

	if(cubin64Bit)
	{
		ELFH32.FileClass = 2;
		ELFH32.Flags |= 0x00000400; //change to 0x0014051x
	}
	else
		ELFH32.FileClass = 1;

	ELFH32.PHTOffset = cubinPHTOffset;
	ELFH32.SHTOffset = ELFHeaderSize;
	

	ELFH32.HeaderSize = ELFHeaderSize;
	ELFH32.PHSize = ELFSegmentHeaderSize;
	ELFH32.PHCount = cubinPHCount;
	ELFH32.SHSize = ELFSectionHeaderSize;
	ELFH32.SHCount = cubinCurrentSectionIndex;
	
	
	
}

unsigned int pad0 = 0;
//Stage7: Write to cubin
void hpCubinWriteSectionHeader(ELFSectionHeader &header, iostream& csOutput)
{
	if(!cubin64Bit)
		csOutput.write((char*)&header, sizeof(ELFSectionHeader));
	else
	{
		csOutput.write((char*)&header,			0x8); //name and type
		csOutput.write((char*)&header.Flags,	0x4); //flags
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.MemImgAddr,0x4); //vaddr
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.FileOffset,0x4); //offset
		csOutput.write((char*)&pad0,			0x4);
		
		csOutput.write((char*)&header.Size,		0x4); //size
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.Link,		0x8);//link and info
		csOutput.write((char*)&header.Alignment,0x4); //alignment
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.EntrySize,0x4); //entry size
		csOutput.write((char*)&pad0,			0x4);
	}
}
void hpCubinWriteSegmentHeader(ELFSegmentHeader &header, iostream& csOutput)
{
	if(!cubin64Bit)
		csOutput.write((char*)&header, sizeof(ELFSegmentHeader));
	else
	{
		csOutput.write((char*)&header.Type,		0x4);//type
		csOutput.write((char*)&header.Flags,	0x4);//flags
		csOutput.write((char*)&header.Offset,	0x4);//offset
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.VirtualMemAddr, 0x4);//vaddr
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.PhysicalMemAddr,0x4);//paddr
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.FileSize, 0x4);//file size
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.MemSize,  0x4);//mem size
		csOutput.write((char*)&pad0,			0x4);
		csOutput.write((char*)&header.Alignment,0x4);//alignment
		csOutput.write((char*)&pad0,			0x4);
	}
}
void hpCubinStage7(iostream& csOutput)
{
	//---Header
	if(!cubin64Bit)
		csOutput.write((char*)&ELFH32, sizeof(ELFH32));
	else
	{
		csOutput.write((char*)&ELFH32,				0x18); //all the way to version
		csOutput.write((char*)&ELFH32.EntryPoint,	0x4);//entry
		csOutput.write((char*)&pad0,				0x4);
		csOutput.write((char*)&ELFH32.PHTOffset,	0x4);//PHTOffset
		csOutput.write((char*)&pad0,				0x4);
		csOutput.write((char*)&ELFH32.SHTOffset,	0x4);//SHTOffset
		csOutput.write((char*)&pad0,				0x4);
		//csOutput.write((char*)&ELFH32.PHTOffset,	0x4);//PHTOffset
		csOutput.write((char*)&ELFH32.Flags,		0x10);//Flags and all the way down

	}

	//---SHT
	//head
	hpCubinWriteSectionHeader(cubinSectionEmpty.SectionHeader, csOutput);
	hpCubinWriteSectionHeader(cubinSectionSHStrTab.SectionHeader, csOutput);
	hpCubinWriteSectionHeader(cubinSectionStrTab.SectionHeader, csOutput);
	hpCubinWriteSectionHeader(cubinSectionSymTab.SectionHeader, csOutput);
	//kern
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		hpCubinWriteSectionHeader(kernel->TextSection.SectionHeader, csOutput);
		hpCubinWriteSectionHeader(kernel->Constant0Section.SectionHeader, csOutput);
		hpCubinWriteSectionHeader(kernel->InfoSection.SectionHeader, csOutput);
		if(kernel->SharedSize != 0)
			hpCubinWriteSectionHeader(kernel->SharedSection.SectionHeader, csOutput);
		if(kernel->LocalSize !=0)
			hpCubinWriteSectionHeader(kernel->LocalSection.SectionHeader, csOutput);
	}
	//tail
	if(cubinConstant2Size)
		hpCubinWriteSectionHeader(cubinSectionConstant2.SectionHeader, csOutput);
	hpCubinWriteSectionHeader(cubinSectionNVInfo.SectionHeader, csOutput);

	//---Sections
	//head
	csOutput.write((char*)cubinSectionSHStrTab.SectionContent, cubinSectionSHStrTab.SectionSize);
	csOutput.write((char*)cubinSectionStrTab.SectionContent, cubinSectionStrTab.SectionSize);
	//..symtab
	if(!cubin64Bit)
		csOutput.write((char*)cubinSectionSymTab.SectionContent, cubinSectionSymTab.SectionSize);
	else
	{
		int entryCount = cubinSectionSymTab.SectionSize / ELFSymbolEntrySize;
		ELFSymbolEntry* entries = (ELFSymbolEntry*)cubinSectionSymTab.SectionContent;
		for(int i =0; i<entryCount; i++)
		{
			csOutput.write((char*)&entries[i].Name, 0x4); //name
			csOutput.write((char*)&entries[i].Info, 0x4); //info, other, SHIndex
			csOutput.write((char*)&entries[i].Value, 0x4); //value
			csOutput.write((char*)&pad0, 0x4);
			csOutput.write((char*)&entries[i].Size, 0x4); //Size
			csOutput.write((char*)&pad0, 0x4);
		}
	}
	//kern
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		csOutput.write((char*)kernel->TextSection.SectionContent, kernel->TextSection.SectionSize);
		csOutput.write((char*)kernel->Constant0Section.SectionContent, kernel->Constant0Section.SectionSize);
		csOutput.write((char*)kernel->InfoSection.SectionContent, kernel->InfoSection.SectionSize);

		delete[] kernel->InfoSection.SectionContent;
		kernel->InfoSection.SectionContent = NULL;
		delete[] kernel->Constant0Section.SectionContent;
		kernel->Constant0Section.SectionContent = NULL;
		delete[] kernel->TextSection.SectionContent;
		kernel->TextSection.SectionContent = NULL;
	}
	//tail
	if(cubinConstant2Size)
		csOutput.write((char*)cubinSectionConstant2.SectionContent, cubinSectionConstant2.SectionSize);
	csOutput.write((char*)cubinSectionNVInfo.SectionContent, cubinSectionNVInfo.SectionSize);

	//---PHT

	hpCubinWriteSegmentHeader(cubinSegmentHeaderPHTSelf, csOutput);
	//kernel segments
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		hpCubinWriteSegmentHeader(kernel->KernelSegmentHeader, csOutput);
		if(kernel->SharedSize || kernel->LocalSize)
			hpCubinWriteSegmentHeader(kernel->MemorySegmentHeader, csOutput);
	}
	//ending segments
	if(cubinConstant2Size)
		hpCubinWriteSegmentHeader(cubinSegmentHeaderConstant2, csOutput);
}
//-----End of cubin helper functions

