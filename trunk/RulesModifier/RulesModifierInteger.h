#ifndef RulesModifierIntegerDefined
#define RulesModifierIntegerDefined


struct ModifierRuleIMUL0U32;
extern ModifierRuleIMUL0U32 MRIMUL0U32;

struct ModifierRuleIMUL1U32;
extern ModifierRuleIMUL1U32 MRIMUL1U32;


struct ModifierRuleIMUL0S32;
extern ModifierRuleIMUL0S32 MRIMUL0S32;

struct ModifierRuleIMUL1S32;
extern ModifierRuleIMUL1S32 MRIMUL1S32;

struct ModifierRuleIMULHI;
extern ModifierRuleIMULHI MRIMULHI;

struct ModifierRuleIMULSAT;
extern ModifierRuleIMULSAT MRIMULSAT;

struct ModifierRuleIADD32ISAT;
extern ModifierRuleIADD32ISAT MRIADD32ISAT;

struct ModifierRuleIADD32IX;
extern ModifierRuleIADD32IX MRIADD32IX;


struct ModifierRuleISETPU32;
extern ModifierRuleISETPU32 MRISETPU32;

struct ModifierRuleVADD_UD;
extern ModifierRuleVADD_UD MRVADD_UD;

struct ModifierRuleVADD_OpType;
extern ModifierRuleVADD_OpType MRVADD_Op1_U8, MRVADD_Op1_U16, MRVADD_Op1_U32, MRVADD_Op1_S8, MRVADD_Op1_S16, MRVADD_Op1_S32;
extern ModifierRuleVADD_OpType MRVADD_Op2_U8, MRVADD_Op2_U16, MRVADD_Op2_U32, MRVADD_Op2_S8, MRVADD_Op2_S16, MRVADD_Op2_S32;

struct ModifierRuleVADD_SAT;
extern ModifierRuleVADD_SAT MRVADD_SAT;

struct ModifierRuleVADD_SecOp;
extern ModifierRuleVADD_SecOp MRVADD_SecOp_MRG_16H, MRVADD_SecOp_MRG_16L, MRVADD_SecOp_MRG_8B0, MRVADD_SecOp_MRG_8B2, MRVADD_SecOp_ACC, MRVADD_SecOp_MIN, MRVADD_SecOp_MAX;

#else
#endif
