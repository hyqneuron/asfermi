#ifndef RulesOperandMemoryDefined
#define RulesOperandMemoryDefined


//Global Memory Operand
struct OperandRuleGlobalMemoryWithImmediate32;
extern OperandRuleGlobalMemoryWithImmediate32 OPRGlobalMemoryWithImmediate32;

//SharedMemory operand
struct OperandRuleSharedMemoryWithImmediate20;
extern OperandRuleSharedMemoryWithImmediate20 OPRSharedMemoryWithImmediate20;



//Constant Memory Operand
struct OperandRuleConstantMemory;
extern OperandRuleConstantMemory OPRConstantMemory;

#else
#endif