#ifndef RulesModifierExecutionDefined
#define RulesModifierExecutionDefined

struct ModifierRuleCALNOINC;
extern ModifierRuleCALNOINC MRCALNOINC;

struct ModifierRuleBRAU;
extern ModifierRuleBRAU MRBRAU;

struct ModifierRuleNOPTRIG;
extern ModifierRuleNOPTRIG MRNOPTRIG;

struct ModifierRuleNOPOP;
extern ModifierRuleNOPOP MRNOPFMA64,
						 MRNOPFMA32,
						 MRNOPXLU  ,
						 MRNOPALU  ,
						 MRNOPAGU  ,
						 MRNOPSU   ,
						 MRNOPFU   ,
						 MRNOPFMUL ;

#endif