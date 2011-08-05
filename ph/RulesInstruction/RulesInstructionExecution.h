#ifndef RulesInstructionExecutionDefined
#define RulesInstructionExecutionDefined


struct InstructionRuleEXIT;
extern InstructionRuleEXIT IREXIT;

struct InstructionRuleCAL;
extern InstructionRuleCAL IRCAL;

struct InstructionRuleBRA;
extern InstructionRuleBRA IRBRA;

struct InstructionRulePRET;
extern InstructionRulePRET IRPRET;

struct InstructionRuleRET;
extern InstructionRuleRET IRRET;

struct InstructionRulePBK;
extern InstructionRulePBK IRPBK;

struct InstructionRuleBRK;
extern InstructionRuleBRK IRBRK;

struct InstructionRulePCNT;
extern InstructionRulePCNT IRPCNT;

struct InstructionRuleCONT;
extern InstructionRuleCONT IRCONT;


struct InstructionRuleNOP;
extern InstructionRuleNOP IRNOP;

#else
#endif