#ifndef RulesOperandCompositeDefined
#define RulesOperandCompositeDefined

struct OperandRuleMOVStyle;
extern OperandRuleMOVStyle OPRMOVStyle;


struct OperandRuleFADDStyle;
extern OperandRuleFADDStyle OPRFADDStyle, OPRFMULStyle;

struct OperandRuleFAllowNegative;
extern OperandRuleFAllowNegative OPRFMULAllowNegative, OPRFFMAAllowNegative;

struct OperandRuleIADDStyle;
extern OperandRuleIADDStyle OPRIADDStyle, OPRIMULStyle;

struct OperandRuleFADDCompositeWithOperator;
extern OperandRuleFADDCompositeWithOperator OPRFADDCompositeWithOperator;

#else
#endif