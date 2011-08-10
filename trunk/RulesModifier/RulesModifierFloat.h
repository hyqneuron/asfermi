#ifndef RulesModifierFloatDefined
#define RulesModifierFloatDefined

struct ModifierRuleFADD32IFTZ;
extern ModifierRuleFADD32IFTZ MRFADD32IFTZ;

struct ModifierRuleFMUL32IFTZ;
extern ModifierRuleFMUL32IFTZ MRFMUL32IFTZ;

struct ModifierRuleFMULR;
extern ModifierRuleFMULR MRFMULRM, MRFMULRP, MRFMULRZ;

struct ModifierRuleFADDSAT;
extern ModifierRuleFADDSAT MRFADDSAT;

struct ModifierRuleFMULSAT;
extern ModifierRuleFMULSAT MRFMULSAT;

struct ModifierRuleMUFU;
extern ModifierRuleMUFU
	MRMUFUCOS,
	MRMUFUSIN,
	MRMUFUEX2,
	MRMUFULG2,
	MRMUFURCP,
	MRMUFURSQ,
	MRMUFURCP64H,
	MRMUFURSQ64H;

#endif