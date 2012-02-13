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

#ifndef CubinDefined 
#define CubinDefined


#include <list>
#include <iostream>

#include "SubString.h"
#include "DataTypes.h"


extern const unsigned int ELFFlagsForsm_20;
extern const unsigned int ELFFlagsForsm_21;

extern int ELFHeaderSize;
extern int ELFSectionHeaderSize;
extern int ELFSegmentHeaderSize;
extern int ELFSymbolEntrySize;

struct ELFHeader32
{
	unsigned char Byte0, Byte1, Byte2, Byte3, FileClass, Encoding, FileVersion, Padding[9];
	unsigned short int FileType, Machine;
	unsigned int Version, EntryPoint, PHTOffset, SHTOffset, Flags;
	unsigned short int HeaderSize, PHSize, PHCount, SHSize, SHCount, SHStrIdx;
	ELFHeader32();
}; //size: 0x34
extern ELFHeader32 ELFH32;


enum SectionType{KernelText, KernelInfo, KernelShared, KernelLocal, KernelConstant0, KernelConstant16, /* Constant2, */ NVInfo, SHStrTab, StrTab, SymTab };
struct ELFSectionHeader
{
	unsigned int NameIndex, Type, Flags, MemImgAddr, FileOffset, Size, Link, Info, Alignment, EntrySize;
}; //size: 0x28
//64 size: 0x40
struct ELFSection
{
	ELFSectionHeader SectionHeader;

	unsigned char* SectionContent;
	unsigned int SectionIndex;
	unsigned int SectionSize;
	unsigned int SHStrTabOffset;
	unsigned int SymbolIndex;
	ELFSection();
};
struct ELFSymbolEntry
{
	unsigned int Name, Value, Size;
	unsigned char Info, Other;
	unsigned short int SHIndex;
	void Reset();
}; //size: 0x10|0x18

struct ELFSegmentHeader
{
	unsigned int Type, Offset, VirtualMemAddr, PhysicalMemAddr, FileSize, MemSize, Flags, Alignment;
}; //size: 0x20
//size: 0x0x38

struct KernelParameter
{
	unsigned int Size;
	unsigned int Offset;
};

struct Kernel
{
	SubString KernelName;
	unsigned int TextSize, SharedSize, LocalSize;
	unsigned int BarCount, RegCount;
	unsigned int StrTabOffset;
	unsigned int MinStackSize, MinFrameSize;
	unsigned int GlobalSymbolIndex;
	
	ELFSection TextSection, Constant0Section, InfoSection, SharedSection, LocalSection;
	ELFSegmentHeader KernelSegmentHeader, MemorySegmentHeader;
	
	std::list<KernelParameter> Parameters;
	unsigned int ParamTotalSize;
	std::list<Instruction> KernelInstructions;

	void Reset();
};

struct Constant2
{
	string Constant2Name;
	unsigned int Offset;
	unsigned int StrTabOffset;
};
#endif
