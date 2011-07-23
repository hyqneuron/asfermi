#include "DataTypes.h"
#include "GlobalVariables.h"
#include "helper\helperException.h"
#include "RulesDirective.h"
#include "SpecificParsers.h"


//Kernel
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

//EndKernel
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

//Param
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
			if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
				count = currentArg->ToImmediate32FromHexConstant(false);
			else
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

//Shared
struct DirectiveRuleShared: DirectiveRule 
{
	DirectiveRuleShared()
	{
		Name = "Shared";
	}
	virtual void Process()//!Local Size 
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		csCurrentKernel.SharedSize = size;
	}
}DRShared;

//Local
struct DirectiveRuleLocal: DirectiveRule 
{
	DirectiveRuleLocal()
	{
		Name = "Local";
	}
	virtual void Process()//!Local Size 
	{
		if(!csCurrentKernelOpened)
			throw 1006; //only definable inside kernels
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		csCurrentKernel.LocalSize = size;
	}
}DRLocal;

//Constant2
struct DirectiveRuleConstant2: DirectiveRule //!Constant2 size
{
	DirectiveRuleConstant2()
	{
		Name = "Constant2";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002; //incorrect no. of arguments
		if(cubinConstant2Size)
			throw 1015; //Constant2 could be declared only once per cubin.

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin(); 
		currentArg++;//currentArg is on size
		int size;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			size = currentArg->ToImmediate32FromHexConstant(false);
		else
			size = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?
		if(size>65536)
			throw 1014; //Maximal constant2 size supported is 65536 bytes.
		cubinConstant2Size = size;
		cubinSectionConstant2.SectionSize = size;
		cubinSectionConstant2.SectionContent = new unsigned char[size];
		memset(cubinSectionConstant2.SectionContent, 0, size);
	}
}DRConstant2;


struct DirectiveRuleConstant: DirectiveRule //!Constant type offset
{
	DirectiveRuleConstant()
	{
		Name = "Constant";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=3)
			throw 1002; //incorrect no. of arguments
		if(cubinConstant2Size==0)
			throw 1012; //constant2 size must be declared to non-zero before constant could be declared

		list<SubString>::iterator currentArg = csCurrentDirective.Parts.begin();

		//type
		currentArg++;
		if(currentArg->Compare("int"))
			cubinCurrentConstant2Parser = &Constant2ParseInt;
		else if(currentArg->Compare("long"))
			cubinCurrentConstant2Parser = &Constant2ParseLong;
		else if(currentArg->Compare("float"))
			cubinCurrentConstant2Parser = &Constant2ParseFloat;
		else if(currentArg->Compare("double"))
			cubinCurrentConstant2Parser = &Constant2ParseDouble;
		else if(currentArg->Compare("mixed"))
			cubinCurrentConstant2Parser = &Constant2ParseMixed;
		else
			throw 1016; //Unsupported constant type


		//offset
		currentArg++;
		int offset;
		if((*currentArg).Length>2 && (*currentArg)[0]=='0' && ((*currentArg)[1]=='x')||(*currentArg)[1]=='X')
			offset = currentArg->ToImmediate32FromHexConstant(false);
		else
			offset = currentArg->ToImmediate32FromInt32(); //issue: what's the error message that it's gonna give?

		if(offset>cubinConstant2Size)
			throw 1013; //Offset is larger than constant2

		cubinCurrentConstant2Offset = offset;
		csLineParserStack.push(csLineParser);
		csLineParser = (LineParser*)&LPConstant2;
	}
}DRConstant;


struct DirectiveRuleEndConstant: DirectiveRule //!Constant type offset
{
	DirectiveRuleEndConstant()
	{
		Name = "EndConstant";
	}
	virtual void Process()
	{
		csLineParser = csLineParserStack.top();
		csLineParserStack.pop();
	}
}DREndConstant;


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


struct DirectiveRuleSelfDebug: DirectiveRule
{
	DirectiveRuleSelfDebug()
	{
		Name = "SelfDebug";
	}
	virtual void Process()
	{
		if(csCurrentDirective.Parts.size()!=2)
			throw 1002;
		list<SubString>::iterator part = csCurrentDirective.Parts.begin();
		part++;
		char zeroSaver = part->Start[part->Length];
		part->Start[part->Length] = 0;
		if(strcmp("On", part->Start)==0)
		{
			csSelfDebug = true;
		}
		else if(strcmp("Off", part->Start)==0)
		{
			csSelfDebug = false;
		}
		else
			throw 0;// unsupported argument
	}
}DRSelfDebug;
