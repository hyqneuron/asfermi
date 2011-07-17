/*
This file contains rules for operands
*/
#ifndef RulesOperandDefined

inline void WriteToImmediate32(unsigned int content)
{
	csCurrentInstruction.OpcodeWord0 |= content <<26;
	csCurrentInstruction.OpcodeWord1 |= content >> 6;
}
inline void MarkConstantMemoryForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 1<<14; //constant memory flag
}
inline void MarkImmediate20ForImmediate32()
{
	csCurrentInstruction.OpcodeWord1 |= 3<<14; //20-bit immediate flag
}
inline void MarkRegisterForImmediate32()
{
}

#include "RulesOperand\RulesOperandConstant.h"
#include "RulesOperand\RulesOperandRegister.h"
#include "RulesOperand\RulesOperandMemory.h"
#include "RulesOperand\RulesOperandComposite.h"
#include "RulesOperand\RulesOperandOthers.h"

//	5.2
//-----Specific Operand Rules
/*
Primary operand types: 
	Type1:	Register, Hex Constant, Predicate Register
	Type2:	Integer Constant, Float Constant, S2R operand
	Type3:	Constant Memory, Global Memory, Constant Memory
	
Primary operand types have their range checking done in SubString member functions

Secondary operand types:
	MOVStyle	:Register, Constant memory without reg, 20-bit Hex, Int, Float)
	FADDStyle	:Register, Constant memory without reg, 20-bit float)
	FIADDStyle	:Register, Constant memory without reg, 20-bit Int)

Secondary operand processors only check that components have the correct prefix and then send components to SubString
	member functions for processing. Primary operand processors then directly write the result to OpcodeWord.
Composite operator processor would sometimes have to check the value returned by SubString functions fall within
	a defined, smaller range before writing.
*/
//SubString member functions only assume a least length of 1








#else
#define RulesOperandDefined
#endif