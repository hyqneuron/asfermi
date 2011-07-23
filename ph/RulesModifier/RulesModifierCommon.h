#ifndef RulesModifierCommonDefined
#define RulesModifierCommonDefined


struct ModifierRuleSETPLogic;
extern ModifierRuleSETPLogic MRSETPLogicAND, MRSETPLogicOR, MRSETPLogicXOR;

struct ModifierRuleSETPComparison;
extern ModifierRuleSETPComparison 
	MRSETPComparisonLT,
	MRSETPComparisonEQ,
	MRSETPComparisonLE,
	MRSETPComparisonGT,
	MRSETPComparisonNE,
	MRSETPComparisonGE,
	MRSETPComparisonNUM,
	MRSETPComparisonNAN,
	MRSETPComparisonLTU,
	MRSETPComparisonEQU,
	MRSETPComparisonLEU,
	MRSETPComparisonGTU,
	MRSETPComparisonNEU,
	MRSETPComparisonGEU;


#else
#endif