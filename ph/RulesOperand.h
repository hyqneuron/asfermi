#include "DataTypes.h"
#include "GlobalVariables.h"


#ifndef RulesOperandDefined
#define RulesOperandDefined

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

#include "RulesOperand/RulesOperandConstant.h"
#include "RulesOperand/RulesOperandRegister.h"
#include "RulesOperand/RulesOperandMemory.h"
#include "RulesOperand/RulesOperandComposite.h"
#include "RulesOperand/RulesOperandOthers.h"

#else
#endif