#ifndef RulesOperandMemoryDefined



//Global Memory Operand
struct OperandRuleGlobalMemoryWithImmediate32: OperandRule
{
	OperandRuleGlobalMemoryWithImmediate32(): OperandRule(GlobalMemoryWithImmediate32){}
	virtual void Process(SubString &component)
	{
		unsigned int memory; int register1;
		component.ToGlobalMemory(register1, memory);
		//Check max reg when register is not RZ(63)
		if(register1!=63)
			csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		WriteToImmediate32(memory);
	}
}OPRGlobalMemoryWithImmediate32;

//SharedMemory operand
struct OperandRuleSharedMemoryWithImmediate20: OperandRule
{
	OperandRuleSharedMemoryWithImmediate20(): OperandRule(SharedMemoryWithImmediate20){}
	virtual void Process(SubString &component)
	{
		unsigned int memory; int register1;
		component.ToGlobalMemory(register1, memory);
		if(memory>=1<<20) //issue: not sure if negative hex is gonna work
			throw 130; //cannot be longer than 20 bits
		//Check max reg when register is not RZ(63)
		if(register1!=63)
			csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		WriteToImmediate32(memory);
	}
}OPRSharedMemoryWithImmediate20;



//Constant Memory Operand
struct OperandRuleConstantMemory: OperandRule
{
	OperandRuleConstantMemory() : OperandRule(ConstantMemory){}
	virtual void Process(SubString &component)
	{		
		unsigned int bank, memory;
		int register1;
		component.ToConstantMemory(bank, register1, memory);
		if(register1!=63)
			csMaxReg = (register1 > csMaxReg)? register1: csMaxReg;

		csCurrentInstruction.OpcodeWord0 |= register1<<20; //RE1
		csCurrentInstruction.OpcodeWord1 |= bank<<10;
		WriteToImmediate32(memory);
		//no need to do the marking for constant memory
	}
}OPRConstantMemory;

#else
#define RulesOperandMemoryDefined
#endif