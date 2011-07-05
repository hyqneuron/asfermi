/*
This file contains various helper functions used during cubin output (the post-processin stage)
1: Stage1 functions
2: Stage2 functions
3: Stage3 functions
4: Stage4 and later functions

all functions are prefixed with 'hpCubin'
*/

#ifndef helperCubinDefined //prevent multiple inclusion
//---code starts ---

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
	case SectionType::KernelText:
		cubinCurrentSHStrTabOffset += 7; //".text." + 0
		break;
	case SectionType::KernelConstant0:
		cubinCurrentSHStrTabOffset += 15; //".nv.constant0." + 0
		break;
	case SectionType::KernelInfo:
		cubinCurrentSHStrTabOffset += 10; //".nv.info."  + 0
		break;
	case SectionType::KernelShared:
		cubinCurrentSHStrTabOffset += 12; //".nv.shared." + 0
		break;
	case SectionType::KernelLocal:
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
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		//Text
		hpCubinStage1SetSection(kernel->TextSection, SectionType::KernelText, kernel->KernelName.Length);
		//Constant0
		hpCubinStage1SetSection(kernel->Constant0Section, SectionType::KernelConstant0, kernel->KernelName.Length);
		//Info
		int infoSize = 12;
		if(kernel->Parameters.size()!=0)
			infoSize = 0x14 * (kernel->Parameters.size()+1);
		hpCubinStage1SetSection(kernel->InfoSection, SectionType::KernelInfo, kernel->KernelName.Length);
		//Shared
		if(kernel->SharedSize!=0)
			hpCubinStage1SetSection(kernel->SharedSection, SectionType::KernelShared, kernel->KernelName.Length);
		//Local
		if(kernel->LocalSize!=0)
			hpCubinStage1SetSection(kernel->LocalSection, SectionType::KernelLocal, kernel->KernelName.Length);
		//StrTaboffset
		kernel->StrTabOffset = cubinCurrentStrTabOffset;
		cubinCurrentStrTabOffset += kernel->KernelName.Length + 1;//increment by length of kernel name + length of endng zero
	}
	//Setup SectionIndex, SHStrTabOffset for .nv.info, nv.constant2
	
	//constant2 not implemented yet
	//cubinSectionConstant2.SectionIndex = cubinCurrentSectionIndex++;
	//cubinSectionConstant2.SHStrTabOffset = cubinCurrentSHStrTabOffset;
	//cubinCurrentSHStrTabOffset += strlen(cubin_str_constant2) + 1;

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
inline void hpCubinAddSectionName2(unsigned char* sectionContent, int &offset, char* sectionName)
{
	memcpy(sectionContent + offset, sectionName, strlen(sectionName)+1);
	offset += strlen(sectionName)+1;
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
	//constant2 not implemented
	hpCubinAddSectionName2(cubinSectionSHStrTab.SectionContent, currentOffset, cubin_str_nvinfo);
}
inline void hpCubinStage2SetStrTabSectionContent()
{
	cubinSectionStrTab.SectionSize = cubinCurrentStrTabOffset;
	cubinSectionStrTab.SectionContent = new unsigned char[cubinCurrentStrTabOffset];

	cubinSectionStrTab.SectionContent[0] = 0;
	int currentOffset = 1;
	//1 entry for each kernel
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		hpCubinAddSectionName3(cubinSectionStrTab.SectionContent, currentOffset, kernel->KernelName);
	}
}

inline void hpCubinStage2AddSectionSymbol(ELFSection &section, ELFSymbolEntry &entry, int &offset, unsigned int size)
{
		entry.Reset();
		entry.Size = size;
		entry.Info = 3;
		entry.SHIndex = section.SectionIndex;
		*(ELFSymbolEntry*)(cubinSectionSymTab.SectionContent + offset) = entry;
		section.SymbolIndex = offset / 0x10;

		offset += 0x10;
}
inline void hpCubinStage2SetSymTabSectionContent()
{		
	int entryCount = cubinCurrentSectionIndex + csKernelList.size() + 2; //1 for each section, 1 for each kernel, 2 empty entries
	cubinSectionSymTab.SectionSize = entryCount * 0x10;
	cubinSectionSymTab.SectionContent = new unsigned char[cubinSectionSymTab.SectionSize];

	//first 6 entries
	memset(cubinSectionSymTab.SectionContent, 0, cubinSectionSymTab.SectionSize); //clear everything to 0 first
	//jump over the entry 0 (null), to directly to entry 1
	//set symbol for head sections
	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x1E) = 1; //only setting section index and info, leaving other things zero
	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x1C) = 3;

	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x2E) = 2;
	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x2C) = 3;

	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x3E) = 3;
	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x3C) = 3;

	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x4C) = 3;
	*(unsigned short int*)(cubinSectionSymTab.SectionContent + 0x5C) = 3;

	//one entry per kern section	
	int offset = 0x60; //jump over entry 4 and 5 which are empty
	ELFSymbolEntry entry;
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		//text
		hpCubinStage2AddSectionSymbol(kernel->TextSection, entry, offset, kernel->TextSize);
		//constant0
		hpCubinStage2AddSectionSymbol(kernel->Constant0Section, entry, offset, 0);
		//info
		hpCubinStage2AddSectionSymbol(kernel->InfoSection, entry, offset, 0);
		//shared section
		if(kernel->SharedSize>0)
			hpCubinStage2AddSectionSymbol(kernel->SharedSection, entry, offset, 0);
		//local section
		if(kernel->LocalSize>0)
			hpCubinStage2AddSectionSymbol(kernel->LocalSection, entry, offset, 0);
	}
	//tail sections
	//constant2 not implemented
	//hpCubinStage2AddSectionSymbol(cubinSectionConstant2, entry, offset, 0);
	hpCubinStage2AddSectionSymbol(cubinSectionNVInfo, entry, offset, 0);

	//one entry per __global__ function
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		entry.Name = kernel->StrTabOffset;
		entry.Value = 0;
		entry.Size = kernel->TextSize;
		entry.Info = 0x12;
		entry.Other = 0x10;
		entry.SHIndex = kernel->TextSection.SectionIndex;
		*(ELFSymbolEntry*)(cubinSectionSymTab.SectionContent + offset) = entry;
		kernel->GlobalSymbolIndex = offset / 0x10;
		offset += 0x10;
	}

	//one entry per constant symbol
	//constant symbol not implemented yet
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
			exception up("assembled kernel size incorrect");
			throw up;
		}

		//.constant0
		kernel->Constant0Section.SectionSize = 0x20 + kernel->ParamTotalSize;
		kernel->Constant0Section.SectionContent = new unsigned char[kernel->Constant0Section.SectionSize];
		memset(kernel->Constant0Section.SectionContent, 0, kernel->Constant0Section.SectionSize); //just set it all to 0

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
			kernel->InfoSection.SectionSize = 0x14 * (kernel->Parameters.size() + 1);//size = (n+1)(0x14)
			kernel->InfoSection.SectionContent = new unsigned char[kernel->InfoSection.SectionSize]; 
			offset = (unsigned int *)kernel->InfoSection.SectionContent;
			//---cbank_param_offsets
			*offset++ = 0x00000c04 | kernel->Parameters.size()*4 << 16; //04 0c aa bb: bbaa is paramcount * 4
			//offset of each argument
			for(list<KernelParameter>::iterator param = kernel->Parameters.begin(); param != kernel->Parameters.end(); param++)
				*offset++ = param->Offset;

			//---param_cbank
			*offset++ = 0x00080a04; //size to follow is always 08
			*offset++ = kernel->Constant0Section.SymbolIndex;
			*offset++ = kernel->ParamTotalSize << 16 | 0x0020; //0x00aa0020: 0xaaaa: total parameter size

			//---cbank_param_size
			*offset++ = 0x00001903 | kernel->ParamTotalSize << 16; //03 19 aa bb: 0xbbaa: total param size

			//---kparam_info
			unsigned int ordinal = kernel->Parameters.size() - 1; //starts from the end of the param list
			for(list<KernelParameter>::reverse_iterator param = kernel->Parameters.rbegin(); param != kernel->Parameters.rend(); param++)
			{
				*offset++ = 0x000c1704; //identifier: 04 17 0c 00
				*offset++ = 0xffffffff; //index, always -0x1
				*offset++ = ( ordinal-- )| (param->Offset<<16); // aa bb cc dd: bbaa is ordinal, ddcc is offset
				*offset++ = 0x0011f000; //issue: only works for param size of 4 bytes
			}
		}

		//.shared and .local have no file space and do not need setup
	}
	

	//tail sections
	//.nv.constant2 not yet supported
	
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

void hpCubinSetELFSectionHeader1(ELFSection &section, unsigned int type, unsigned int alignment, unsigned int &offset)
{
	memset(&section.SectionHeader, 0, sizeof(ELFSectionHeader));
	section.SectionHeader.NameIndex = section.SHStrTabOffset;
	section.SectionHeader.Type = type;
	//Flags remains 0
	//MemImgAddr remains 0
	section.SectionHeader.FileOffset = offset;
	section.SectionHeader.Size = section.SectionSize;
	//Link, Info remains 0
	section.SectionHeader.Alignment = alignment;
	//Entry size remains 0
	offset += section.SectionSize;
}

//Stage4: Setup all section headers
void hpCubinStage4()
{
	unsigned int fileOffset = 0x34 + 0x28 * cubinCurrentSectionIndex; //start of the shstrtab section content

	//---head sections
	//empty
	memset(&cubinSectionEmpty.SectionHeader, 0, sizeof(ELFSectionHeader));
	//shstrtab
	hpCubinSetELFSectionHeader1(cubinSectionSHStrTab, 3, 4, fileOffset);
	//strtab
	hpCubinSetELFSectionHeader1(cubinSectionStrTab, 3, 1, fileOffset);
	//symtab
	hpCubinSetELFSectionHeader1(cubinSectionSymTab, 2, 1, fileOffset);
	cubinSectionSymTab.SectionHeader.EntrySize = 0x10;
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
			hpCubinSetELFSectionHeader1(kernel->SharedSection, 8, 4, fileOffset);
			kernel->SharedSection.SectionHeader.Flags = 3;
			kernel->SharedSection.SectionHeader.Info = kernel->TextSection.SectionIndex;
		}
		//.local
		if(kernel->LocalSize>0)
		{
			hpCubinSetELFSectionHeader1(kernel->LocalSection, 8, 4, fileOffset);
			kernel->LocalSection.SectionHeader.Flags = 3;
			kernel->LocalSection.SectionHeader.Info = kernel->TextSection.SectionIndex;
		}
	}
	//---tail sections
	//constant2 not supported yet
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

	
	cubinPHCount = count + 1; // +1 is the SELF. should +2 if constant2 is to be included
	//PHTSelf
	cubinSegmentHeaderPHTSelf.Type = 6;
	cubinSegmentHeaderPHTSelf.Flags = 5;
	cubinSegmentHeaderPHTSelf.Alignment = 4;
	cubinSegmentHeaderPHTSelf.Offset = cubinPHTOffset;
	cubinSegmentHeaderPHTSelf.FileSize = 0x20 * cubinPHCount;
	cubinSegmentHeaderPHTSelf.MemSize = cubinSegmentHeaderPHTSelf.FileSize;
	cubinSegmentHeaderPHTSelf.VirtualMemAddr = 0;
	cubinSegmentHeaderPHTSelf.PhysicalMemAddr = 0;
	//constant2 not implemented
}

//Stage6: Setup ELF header
void hpCubinStage6()
{
	ELFHeader32.PHTOffset = cubinPHTOffset;
	ELFHeader32.PHCount = cubinPHCount;
	ELFHeader32.SHCount = cubinCurrentSectionIndex;
	if(cubinArchitecture == sm_20)
		ELFHeader32.Flags = ELFFlagsForsm_20;
	else //issue: supports only sm_20 and sm_21
		ELFHeader32.Flags = ELFFlagsForsm_21;
}

//Stage7: Write to cubin
void hpCubinStage7()
{
	//---Header
	csOutput.write((char*)&ELFHeader32, sizeof(ELFHeader));

	//---SHT
	//head
	csOutput.write((char*)&cubinSectionEmpty.SectionHeader, 0x28);
	csOutput.write((char*)&cubinSectionSHStrTab.SectionHeader, 0x28);
	csOutput.write((char*)&cubinSectionStrTab.SectionHeader, 0x28);
	csOutput.write((char*)&cubinSectionSymTab.SectionHeader, 0x28);
	//kern
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		csOutput.write((char*)&kernel->TextSection.SectionHeader, 0x28);
		csOutput.write((char*)&kernel->Constant0Section.SectionHeader, 0x28);
		csOutput.write((char*)&kernel->InfoSection.SectionHeader, 0x28);
		if(kernel->SharedSize != 0)
			csOutput.write((char*)&kernel->TextSection.SectionHeader, 0x28);
		if(kernel->LocalSize !=0)
			csOutput.write((char*)&kernel->TextSection.SectionHeader, 0x28);
	}
	//tail
	//constant2 not implemented
	csOutput.write((char*)&cubinSectionNVInfo.SectionHeader, 0x28);

	//---Sections
	//head
	csOutput.write((char*)cubinSectionSHStrTab.SectionContent, cubinSectionSHStrTab.SectionSize);
	csOutput.write((char*)cubinSectionStrTab.SectionContent, cubinSectionStrTab.SectionSize);
	csOutput.write((char*)cubinSectionSymTab.SectionContent, cubinSectionSymTab.SectionSize);
	//kern
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		csOutput.write((char*)kernel->TextSection.SectionContent, kernel->TextSection.SectionSize);
		csOutput.write((char*)kernel->Constant0Section.SectionContent, kernel->Constant0Section.SectionSize);
		csOutput.write((char*)kernel->InfoSection.SectionContent, kernel->InfoSection.SectionSize);
		if(kernel->SharedSize != 0)
			csOutput.write((char*)kernel->SharedSection.SectionContent, kernel->SharedSection.SectionSize);
		if(kernel->LocalSize !=0)
			csOutput.write((char*)kernel->LocalSection.SectionContent, kernel->LocalSection.SectionSize);
	}
	//tail
	csOutput.write((char*)cubinSectionNVInfo.SectionContent, cubinSectionNVInfo.SectionSize);

	//---PHT
	csOutput.write((char*)&cubinSegmentHeaderPHTSelf, 0x20);
	//kernel segments
	for(list<Kernel>::iterator kernel = csKernelList.begin(); kernel != csKernelList.end(); kernel++)
	{
		csOutput.write((char*)&kernel->KernelSegmentHeader, 0x20);
		if(kernel->SharedSize || kernel->LocalSize)
			csOutput.write((char*)&kernel->MemorySegmentHeader, 0x20);
	}
	//ending segments
	//constant2 not implemented

	//end
	csOutput.flush();
	csOutput.close();
}
//-----End of cubin helper functions



#else
#define helperCubinDefined
#endif