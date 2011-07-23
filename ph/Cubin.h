#ifndef CubinDefined 
#define CubinDefined


#include <list>
#include "SubString.h"
#include "DataTypes.h"


extern const unsigned int ELFFlagsForsm_20;
extern const unsigned int ELFFlagsForsm_21;

struct ELFHeader32
{
	unsigned char Byte0, Byte1, Byte2, Byte3, FileClass, Encoding, FileVersion, Padding[9];
	unsigned short int FileType, Machine;
	unsigned int Version, EntryPoint, PHTOffset, SHTOffset, Flags;
	unsigned short int HeaderSize, PHSize, PHCount, SHSize, SHCount, SHStrIdx;
	ELFHeader32();
};
extern ELFHeader32 ELFH32;

struct ELFHeader64
{
	unsigned char Byte0, Byte1, Byte2, Byte3, FileClass, Encoding, FileVersion, Padding[9];
	unsigned short int FileType, Machine;
	unsigned int Version;
	unsigned long long EntryPoint, PHTOffset, SHTOffset;
	unsigned int Flags;
	unsigned short int HeaderSize, PHSize, PHCount, SHSize, SHCount, SHStrIdx;
	ELFHeader64();
};
extern ELFHeader64 ELFH64;

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
	ELFSection();
};
struct ELFSymbolEntry
{
	unsigned int Name, Value, Size;
	unsigned char Info, Other;
	unsigned short int SHIndex;
	void Reset();
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
	
	std::list<KernelParameter> Parameters;
	unsigned int ParamTotalSize;
	std::list<Instruction> KernelInstructions;

	void Reset();
};
#else
#endif