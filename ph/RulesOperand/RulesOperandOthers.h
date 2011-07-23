#ifndef RulesOperandOthersDefined
#define RulesOperandOthersDefined


//ignored operand: currently used for NOP
struct OperandRuleIgnored;
extern OperandRuleIgnored OPRIgnored;


struct OperandRule32I;
extern OperandRule32I OPR32I;

struct OperandRuleLOP;
extern OperandRuleLOP OPRLOP1, OPRLOP2;


struct OperandRuleF2I;
extern OperandRuleF2I OPRF2I, OPRI2F;


#else
#endif