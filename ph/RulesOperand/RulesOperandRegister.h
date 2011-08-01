#ifndef RulesOperandRegisterDefined
#define RulesOperandRegisterDefined


struct OperandRuleRegister;
extern OperandRuleRegister OPRRegister0, OPRRegister1, OPRRegister2;

//reg3 used a separate rule because it applies it result to OpcodeWord1 instead of 0
struct OperandRuleRegister3;
extern OperandRuleRegister3 OPRRegister3ForMAD, OPRRegister3ForCMP;

//Note that some operands can have modifiers
//This rule deals with registers that can have the .CC modifier
struct OperandRuleRegisterWithCC;
extern OperandRuleRegisterWithCC OPRRegisterWithCC4IADD32I, OPRRegisterWithCCAt16;


//Predicate register operand
struct OperandRulePredicate;
extern OperandRulePredicate OPRPredicate1, OPRPredicate0, OPRPredicate2NotNegatable;

//Some predicate registers expressions can be negated with !
//this kind of operand is processed separately
struct OperandRulePredicate2;
extern OperandRulePredicate2 OPRPredicate2;


struct OperandRuleFADD32IReg1;
extern OperandRuleFADD32IReg1 OPRFADD32IReg1;

struct OperandRuleRegister1WithSignFlag;
extern OperandRuleRegister1WithSignFlag OPRIMADReg1, OPRISCADDReg1;

#endif