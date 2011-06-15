#if defined helperExceptionDefined //prevent multiple inclusion
#else
#define helperExceptionDefined yes
//---code starts ---
//#include <vld.h> //remove when you compile


#include "DataTypes.h"
#include "GlobalVariables.h"
#include "SpecificRules.h"

extern void hpUsage();

void hpExceptionHandler(int e)		//===
{
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
	case 20:	message = "Insufficient number of arguments";
		break;
	case 50:	message = "Initialization error. Repeating instruction indices.";
		break;
	case 99:	message = "Not in replace mode.";
		break;
	default:	message = "No error message";
		break;
	};
	cout<<message<<endl;
	if(csExceptionPrintUsage)
		hpUsage();
}
void hpErrorHandler(int e)
{
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
	default:	message = "Unknown Error";
		break;
	};
	char *line = csCurrentInstruction->InstructionString.ToCharArrayStopOnCR();
	cout<<"Error Line "<<csCurrentInstruction->LineNumber<<": "<<line<<": "<<message<<endl;
	delete[] line;
}
void hpWarning(int e)
{
	char* message;
	switch(e)
	{
	case 10:	message = "Evaluation of constant returned zero. Consider using RZ if value of zero is intended.";
		break;
	case 11:	message = "Evaluation of constant overflowed.";
		break;
	default:	message = "No warning message available.";
		break;
	}
	char *line = csCurrentInstruction->InstructionString.ToCharArrayStopOnCR();
	cout<<"Warning Line "<<csCurrentInstruction->LineNumber<<": "<<line<<": "<<message<<endl;
	delete[] line;
}


#endif