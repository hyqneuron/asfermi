/*
This file contains structures used for cubin output: ELFHeader, ELFSectionHeader, ELFProgramHeader, ELFSymbolEntry
*/
#ifndef CubinDefined  //prevent multiple inclusion
//---code starts ---


struct ELFHeader
{
	unsigned char Byte0, Byte1, Byte2, Byte3, FileClass, Encoding, FileVersion, Padding[9];
	unsigned short int FileType, Machine;
	unsigned int Version, EntryPoint, PHTOffset, SHTOffset, Flags;
	unsigned short int HeaderSize, PHSize, PHCount, SHSize, SHCount, SHStrIdx;
	ELFHeader(bool Bit_32)
	{
		//0x00
		Byte0 = 0x7f;
		Byte1 = 'E';
		Byte2 = 'L';
		Byte3 = 'F';
		//0x04
		FileClass = Bit_32? 1:2; //1 is 32-bit
		Encoding = 1; //LSB
		FileVersion = 1; //1 for current				
		//0x07
		memset(&Padding, 0, 9);
		Padding[0] = 0x33; //issue: same for all? any 
		Padding[1] = 0x04;

		//0x10
		FileType = 0x0002;
		Machine = 0x00BE;								
		
		//0x14
		Version = 1;
		EntryPoint = 0;
		//PHTOffset not set
		SHTOffset = 0x34;

		//0x24
		Flags = 0x00140000 | 0x0115; //issue: doesn not yet support sm_20 which has a value of | ox0114
		//0x0014 is the ptx target architecture
		
		//0x28
		HeaderSize = 0x34;
		PHSize = 0x20;
		//PHCount not set
		SHSize = 0x28;
		//SHCount not set;
		SHStrIdx = 1;									//0x34
	}
}ELFHeader32(true), ELFHeader64(false);

enum SectionType{KernelText, KernelInfo, KernelShared, KernelLocal, KernelConstant0, KernelConstant16, Constant2, NVInfo, SHStrTab, StrTab, SymTab };
struct ELFSectionHeader
{
	unsigned int NameIndex, Type, Flags, MemImgAddr, FileOffset, Size, Link, Info, Alignment, EntrySize;
};

struct ELFSection
{
	ELFSectionHeader SectionHeader;

	unsigned char* SectionContent;
	unsigned int SectionIndex;
	unsigned int SectionSize;
	unsigned int SHStrTabOffset;
	unsigned int SymbolIndex;
};
struct ELFSymbolEntry
{
	unsigned int Name, Value, Size;
	unsigned char Info, Other;
	unsigned short int SHIndex;
	void Reset()
	{
		Name = 0;
		Value = 0;
		Size = 0;
		Info = 0;
		Other = 0;
		SHIndex = 0;
	}
};

struct ELFSegmentHeader
{
	unsigned int Type, Offset, VirtualMemAddr, PhysicalMemAddr, FileSize, MemSize, Flags, Alignment;
};

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
	
	list<KernelParameter> Parameters;
	unsigned int ParamTotalSize;
	list<Instruction> KernelInstructions;

	void Reset()
	{
		KernelName.Start = 0;
		KernelName.Length = 0;

		TextSize = 0;
		SharedSize = 0;
		LocalSize = 0;
		BarCount = 0;
		RegCount = 0;

		StrTabOffset = 0;
		MinStackSize = 0;
		MinFrameSize = 0;
		GlobalSymbolIndex;

		ParamTotalSize = 0;
		Parameters.clear(); //issue: would this affect the saved kernel?
		KernelInstructions.clear();
	}
};

#else
#define CubinDefined yes
#endif