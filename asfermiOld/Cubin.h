/*
This file contains structures used for cubin output: ELFHeader, ELFSectionHeader, ELFProgramHeader, ELFSymbolEntry
*/
#ifndef CubinDefined  //prevent multiple inclusion
//---code starts ---


const unsigned int ELFFlagsForsm_20 = 0x00140114;
const unsigned int ELFFlagsForsm_21 = 0x00140115;
struct ELFHeader32
{
	unsigned char Byte0, Byte1, Byte2, Byte3, FileClass, Encoding, FileVersion, Padding[9];
	unsigned short int FileType, Machine;
	unsigned int Version, EntryPoint, PHTOffset, SHTOffset, Flags;
	unsigned short int HeaderSize, PHSize, PHCount, SHSize, SHCount, SHStrIdx;
	ELFHeader32()
	{
		//0x00
		Byte0 = 0x7f;
		Byte1 = 'E';
		Byte2 = 'L';
		Byte3 = 'F';
		//0x04
		FileClass = 1; //1 is 32-bit
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
		Flags = 0x00140114; //default sm_20
		//0x0014 is the ptx target architecture
		
		//0x28
		HeaderSize = 0x34;
		PHSize = 0x20;
		//PHCount not set
		SHSize = 0x28;
		//SHCount not set;
		SHStrIdx = 1;									//0x34
	}
}ELFH32;

struct ELFHeader64
{
	unsigned char Byte0, Byte1, Byte2, Byte3, FileClass, Encoding, FileVersion, Padding[9];
	unsigned short int FileType, Machine;
	unsigned int Version;
	unsigned long long EntryPoint, PHTOffset, SHTOffset;
	unsigned int Flags;
	unsigned short int HeaderSize, PHSize, PHCount, SHSize, SHCount, SHStrIdx;
	ELFHeader64()
	{
		Byte0 = 0x7f;
		Byte1 = 'E';
		Byte2 = 'L';
		Byte3 = 'F';

		FileClass = 2; //2 is 64-bit
		Encoding = 1; //LSB
		FileVersion = 1; //1 for current				

		memset(&Padding, 0, 9);
		Padding[0] = 0x33; //issue: same for all? any 
		Padding[1] = 0x04;

		FileType = 0x0002;
		Machine = 0x00BE;								
		

		Version = 1;
		EntryPoint = 0;
		//PHTOffset not set
		SHTOffset = 0x40;


		Flags = 0x00140114;
		
		
		HeaderSize = 0x40;
		PHSize = 0x38;
		//PHCount not set
		SHSize = 0x40;
		//SHCount not set;
		SHStrIdx = 1;
	}
}ELFH64;

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
	ELFSection()
	{
		SectionContent = 0;
	}
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