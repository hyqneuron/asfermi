#ifndef RulesDirectiveDefined
#define RulesDirectiveDefined



//Kernel
struct DirectiveRuleKernel;
extern DirectiveRuleKernel DRKernel;

//EndKernel
struct DirectiveRuleEndKernel;
extern DirectiveRuleEndKernel DREndKernel;

//Param
struct DirectiveRuleParam;
extern DirectiveRuleParam DRParam;

//Shared
struct DirectiveRuleShared;
extern DirectiveRuleShared DRShared;

//Local
struct DirectiveRuleLocal;
extern DirectiveRuleLocal DRLocal;

//Constant2
struct DirectiveRuleConstant2;
extern DirectiveRuleConstant2 DRConstant2;


struct DirectiveRuleConstant;
extern DirectiveRuleConstant DRConstant;


struct DirectiveRuleEndConstant;
extern DirectiveRuleEndConstant DREndConstant;


struct DirectiveRuleArch;
extern DirectiveRuleArch DRArch;


struct DirectiveRuleSelfDebug;
extern DirectiveRuleSelfDebug DRSelfDebug;

#else
#endif