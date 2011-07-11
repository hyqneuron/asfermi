/*
This file contains the error/warning handling code.
*/

#ifndef helperExceptionDefined //prevent multiple inclusion
//---code starts ---



extern void hpUsage();

void hpExceptionHandler(int e)		//===
{
	if(csSelfDebug)
		throw exception();
	char *message;
	switch(e)
	{
	case 0:		message = "Invalid arguments.";
		break;
	case 1:		message = "Unable to open input file";
		break;
	case 3:		message = "Incorrect kernel offset.";
		break;
	case 4:		message = "Cannot open output file";
		break;
	case 5:		message = "Cannot find specified kernel";
		break;
	case 6:		message = "File to be modified is invalid.";
		break;
	case 7:		message = "Failed to read cubin";
		break;
	case 8:		message = "Cannot find the specified section";
		break;
	case 9:		message = "Specific section not large enough to contain all the assembled opcodes.";
		break;
	case 10:	message = "Single-line mode only supported in Replace Mode.";
		break;
	case 20:	message = "Insufficient number of arguments";
		break;
	case 50:	message = "Initialization error. Repeating rule indices. Please debug to find out the exact repeating entry.";
		break;
	case 97:	message = "Operation mode not supported.";
		break;
	case 98:	message = "Cannot proceed due to errors.";
		break;
	case 99:	message = "Command-line argument used is not supported.";
		break;
	case 100:	message = "Last kernel is not ended with EndKernel directive.";
		break;
	case 101:	message = "No valid kernel is found in file.";
		break;
	default:	message = "No error message";
		break;
	};
	cout<<message<<endl;
	if(csExceptionPrintUsage)
		hpUsage();
	getchar();
}
void hpInstructionErrorHandler(int e)
{
	if(csSelfDebug)
		throw exception();
	csErrorPresent = true;
	char *message;
	switch(e)
	{
	case 100:	message = "Instruction name is absent following the predicate";
		break;
	case 101:	message = "Unsupported modifier.";
		break;
	case 102:	message = "Too many operands.";
		break;
	case 103:	message = "Insufficient number of operands.";
		break;
	case 104:	message = "Incorrect register format.";
		break;
	case 105:	message = "Register number too large.";
		break;
	case 106:	message = "Incorrect hex value.";
		break;
	case 107:	message = "Incorrect global memory format.";
		break;
	case 108:	message = "Instruction not supported.";
		break;
	case 109:	message = "Incorrect predicate.";
		break;
	case 110:	message = "Incorrect constant memory format.";
		break;
	case 111:	message = "Memory address for constant memory too large.";
		break;
	case 112:	message = "Register cannot be used in MOV-style constant address. Consider using LDC instead.";
		break;
	case 113:	message = "The immediate value is limited to 20-bit.";
		break;
	case 114:	message = "Constant memory bank number too large.";
		break;
	case 115:	message = "Immediate value is limited to 20-bit.";
		break;
	case 116:	message = "Invalid operand.";
		break;
	case 117:	message = "Incorrect floating number.";
		break;
	case 118:	message = "20-bit immediate value cannot contain 64-bit number.";
		break;
	case 119:	message = "Register cannot be used in FADD-style constant address.";
		break;
	case 120:	message = "Only constants can be negative for MOV-style operand.";
		break;
	case 121:	message = "Incorrect floating point number format.";
		break;
	case 122:	message = "Too many modifiers present.";
		break;
	case 123:	message = "Empty modifier.";
		break;
	case 124:	message = "Instruction name not present.";
		break;
	case 125:	message = "Empty operand.";
		break;
	case 126:	message = "Incorrect predicate.";
		break;
	case 127:	message = "Insufficient number of modifiers";
		break;
	case 128:	message = "Incorrect special register name.";
		break;
	case 129:	message = "Negative sign cannot be used here.";
		break;
	default:	message = "Unknown Error.";
		break;
	};
	char *line = csCurrentInstruction.InstructionString.ToCharArrayStopOnCR();
	cout<<"Error Line "<<csCurrentInstruction.LineNumber<<": "<<line<<": "<<message<<endl;
	delete[] line;
}
void hpDirectiveErrorHandler(int e)
{
	if(csSelfDebug)
		throw exception();
	csErrorPresent = true;
	char *message;
	switch(e)
	{
	case 1000:	message = "Empty Instruction.";
		break;
	case 1001:	message = "Unsupported directive.";
		break;
	case 1002:	message = "Incorrect number of directive arguments.";
		break;
	case 1003:	message = "Kernel directive can only be used in DirectOutput Mode.";
		break;
	case 1004:	message = "Without an EndKernel directive, instructions in the previous kernel will be ignored.";
		break;
	case 1005:	message = "Corresponding Kernel directive not found.";
		break;
	case 1006:	message = "Parameters can only be defined within kernel sections.";
		break;
	case 1007:	message = "Size of parameter must be multiples of 4.";
		break;
	case 1008:	message = "Size of parameters cannot exceed 256 bytes.";
		break;
	case 1009:	message = "Parameter count too large.";
		break;
	case 1010:	message = "Incorrect number of kernel arguments.";
		break;
	case 1011:	message = "Unsupported architecture. Please use only sm_20 or sm_21"; //issue: architecture support may increase in future
	default:	message = "Unknown error.";
		break;
	};
	
	char *line = csCurrentDirective.DirectiveString.ToCharArrayStopOnCR();
	cout<<"Error Line "<<csCurrentDirective.LineNumber<<": "<<line<<": "<<message<<endl;
	delete[] line;
}

void hpWarning(int e)
{
	if(csSelfDebug)
		throw exception();
	char* message;
	switch(e)
	{
	case 10:	message = "Evaluation of constant returned zero. Consider using RZ if value of zero is intended.";
		break;
	case 11:	message = "Evaluation of constant overflowed.";
		break;
	case 12:	message = "Some instructions before the kernel section are not included in any kernel.";
		break;
	default:	message = "No warning message available.";
		break;
	}
	char *line = csCurrentInstruction.InstructionString.ToCharArrayStopOnCR();
	cout<<"Warning Line "<<csCurrentInstruction.LineNumber<<": "<<line<<": "<<message<<endl;
	delete[] line;
}

#else
#define helperExceptionDefined
#endif