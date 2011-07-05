/*
This file contains specific rules for various directives.
*/

#ifndef SpecificDirectiveRulesDefined //prevent multiple inclusion
//---code starts ---


struct DirectiveRuleKernel: DirectiveRule
{
	DirectiveRuleKernel()
	{
		Name = "Kernel";
	}
	virtual void Process() // Needs 3 arguments: Name
	{
		if(csOperationMode != DirectOutput)
			throw 1003; //only usable in direct output mode
		if(csCurrentDirective.Parts.size()!=2) //!Kernel KernelName
			throw 1002; //Incorrect number of directive arguments.
		if(csCurrentKernelOpened)
			throw 1004; //previous kernel without EndKernel
		if(csInstructions.size()!=0)
			hpWarning(12); //some instructions not included

		csCurrentKernelOpened = true;
		if(csCurrentDirective.Parts.size()!=2)
			throw 1010; //Incorrect number of directive parameters
		csCurrentKernel.KernelName = *csCurrentDirective.Parts.rbegin();
	}
}DRKernel;
struct DirectiveRuleEndKernel: DirectiveRule
{
	DirectiveRuleEndKernel()
	{
		Name = "EndKernel";
	}
	virtual void Process()
	{
		if(!csCurrentKernelOpened)
			throw 1005; //without Kernel directive
		
		csCurrentKernel.KernelInstructions = csInstructions;
		csCurrentKernel.TextSize = csInstructionOffset;
		csCurrentKernel.RegCount = csMaxReg + 1;
		csKernelList.push_back(csCurrentKernel);

		csInstructions.clear();
		csInstructionOffset = 0;
		csCurrentKernel.Reset();
		csCurrentKernelOpened = false;
	}
}DREndKernel;
struct DirectiveRuleParam: DirectiveRule //!Param Size Count
{
	DirectiveRuleParam()
	{
		Name = "Param";
	}
	virtual void Process()
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()<2 || csCurrentDirective.Parts.size()>3)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		unsigned int size = currentArg->ToImmediate32FromInt32();
		if(size%4 !=0 )
			throw 1007; //size of parameter must be multiple of 4; issue: may not be necessary
		if(size>256)
			throw 1008; //size of parameter cannot be larger than 256

		unsigned int count = 1;
		if(csCurrentDirective.Parts.size()==3)
		{
			currentArg++;
			count = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		}

		if(count>256)
			throw 1009; //prevent overflow

		for(int i =0; i<count; i++)
		{
			KernelParameter param;
			param.Size = size;
			param.Offset = csCurrentKernel.ParamTotalSize;
			csCurrentKernel.ParamTotalSize += size;
			csCurrentKernel.Parameters.push_back(param);
		}
		if(csCurrentKernel.ParamTotalSize > 256)
			throw 1008;		
	}
}DRParam;

struct DirectiveRuleArch: DirectiveRule
{
	DirectiveRuleArch()
	{
		Name = "Arch";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002;
		list<SubString>::iterator part = csCurrentDirective.Parts.begin();
		part++;
		char zeroSaver = part->Start[part->Length];
		part->Start[part->Length] = 0;
		if(strcmp("sm_20", part->Start)==0)
		{
			cubinArchitecture = sm_20;
		}
		else if(strcmp("sm_21", part->Start)==0)
		{
			cubinArchitecture = sm_21;
		}
		else
			throw 0;// unsupported argument
	}
}DRArch;

#else
#define SpecificDirectiveRulesDefined yes
#endif