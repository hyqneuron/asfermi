#include "../DataTypes.h"
#include "../helper/helperMixed.h"

#include "../stdafx.h"
#include "stdafx.h" //SMark

#include "RulesInstructionInteger.h"
#include "../RulesModifier.h"
#include "../RulesOperand.h"



struct InstructionRuleIADD: InstructionRule
{
	InstructionRuleIADD() : InstructionRule("IADD", 2, true, false)
	{
		hpBinaryStringToOpcode8("1100000000111000000000000000000000000000000000000000000000010010", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegisterWithCCAt16,
					&OPRIMADReg1,
					&OPRIADDStyle);
		ModifierGroups[0].Initialize(true, 1, &MRIADD32ISAT);
		ModifierGroups[1].Initialize(true, 1, &MRIADD32IX);
	}
}IRIADD;


struct INstructionRuleIADD32I: InstructionRule
{
	INstructionRuleIADD32I(): InstructionRule("IADD32I", 2, true, false)
	{
		hpBinaryStringToOpcode8("0100000000111000000000000000000000000000000000000000000000010000", OpcodeWord0, OpcodeWord1);
		SetOperands(3, 
					&OPRRegisterWithCC4IADD32I,
          				&OPRRegister1,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 1, &MRIADD32ISAT);
		ModifierGroups[1].Initialize(true, 1, &MRIADD32IX);
	}
}IRIADD32I;


struct InstructionRuleIMUL: InstructionRule
{
	InstructionRuleIMUL(): InstructionRule("IMUL", 3, true, false)
	{
		hpBinaryStringToOpcode8("1100010100111000000000000000000000000000000000000000000000001010", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegisterWithCCAt16,
					&OPRRegister1,
					&OPRIMULStyle);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
	}
}IRIMUL;

struct InstructionRuleIMUL32I: InstructionRule
{
	InstructionRuleIMUL32I() : InstructionRule("IMUL32I", 3, true, false)
	{
		hpBinaryStringToOpcode8("0100 010100 1110 000000 000000 00000000000000000000000000000000 0 01000", OpcodeWord0, OpcodeWord1);
		SetOperands(3,
					&OPRRegisterWithCC4IADD32I, //different cc pos
					&OPRRegister1,
					&OPR32I);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
	}
}IRIMUL32I;

struct InstructionRuleIMAD: InstructionRule
{
	InstructionRuleIMAD(): InstructionRule("IMAD", 4, true, false)
	{
		hpBinaryStringToOpcode8("1100 010100111000000000000000000000000000000000000000000000 000100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegisterWithCCAt16,
					&OPRIMADReg1,
					&OPRIMULStyle,
					&OPRMAD3);
		ModifierGroups[0].Initialize(true, 2, &MRIMUL0U32, &MRIMUL0S32);
		ModifierGroups[1].Initialize(true, 2, &MRIMUL1U32, &MRIMUL1S32);
		ModifierGroups[2].Initialize(true, 1, &MRIMULHI);
		ModifierGroups[3].Initialize(true, 1, &MRIMULSAT);
	}
}IRIMAD;

struct InstructionRuleISCADD: InstructionRule
{
	InstructionRuleISCADD(): InstructionRule("ISCADD", 0, true, false)
	{
		hpBinaryStringToOpcode8("1100 0 00000 1110 000000 000000 0000000000000000000000 0000000000 000010", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegisterWithCCAt16,
					&OPRISCADDReg1,
					&OPRISCADDAllowNegative,
					&OPRISCADDShift);
	}
}IRISCADD;

struct InstructionRuleISETP: InstructionRule
{
	InstructionRuleISETP(): InstructionRule("ISETP", 3, true, false)
	{
		hpBinaryStringToOpcode8("1100 010000 1110 111 000 000000 0000000000000000000000 0 1110 000000 11000", OpcodeWord0, OpcodeWord1);
		SetOperands(5, 
					&OPRPredicate0,
					&OPRPredicate1,
					&OPRRegister1,
					&OPRIADDStyle,
					&OPRPredicate2);
		ModifierGroups[0].Initialize(false, 6, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE);
		ModifierGroups[1].Initialize(true, 1, &MRISETPU32);
		ModifierGroups[2].Initialize(true, 3,
					&MRSETPLogicAND,
					&MRSETPLogicOR,
					&MRSETPLogicXOR);
	}
}IRISETP;

struct InstructionRuleICMP: InstructionRule
{
	InstructionRuleICMP(): InstructionRule("ICMP", 2, true, false)
	{
		hpBinaryStringToOpcode8("1100 010000 1110 000000 000000 0000000000000000000000 0 000000  000 001100", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegister0,
					&OPRRegister1,
					&OPRIMULStyle,
					&OPRRegister3ForCMP);
					
		ModifierGroups[0].Initialize(false, 6, 
					&MRSETPComparisonLT,
					&MRSETPComparisonEQ,
					&MRSETPComparisonLE,
					&MRSETPComparisonGT,
					&MRSETPComparisonNE,
					&MRSETPComparisonGE);
		ModifierGroups[1].Initialize(true, 1, &MRIMUL1U32);
	}
}IRICMP;

struct InstructionRuleVADD: InstructionRule
{
	InstructionRuleVADD(): InstructionRule("VADD", 6, true, false)
	{
		hpBinaryStringToOpcode8("0010 011000 1110 000000 000000 0000000110000000 1001100 000000 111 000011", OpcodeWord0, OpcodeWord1);
		SetOperands(4,
					&OPRRegisterWithCCAt16,
					&OPRRegister1ForVADD,
					&OPRCompositeForVADD,
					&OPRRegister3ForCMP);
		ModifierGroups[0].Initialize(true, 1, &MRVADD_UD);
		ModifierGroups[1].Initialize(true, 6,  
					&MRVADD_Op1_U8,
					&MRVADD_Op1_U16,
					&MRVADD_Op1_U32,
					&MRVADD_Op1_S8,
					&MRVADD_Op1_S16,
					&MRVADD_Op1_S32);
		ModifierGroups[2].Initialize(true, 6, 
					&MRVADD_Op2_U8,
					&MRVADD_Op2_U16,
					&MRVADD_Op2_U32,
					&MRVADD_Op2_S8,
					&MRVADD_Op2_S16,
					&MRVADD_Op2_S32);
		ModifierGroups[3].Initialize(true, 1, &MRVADD_SAT);
		ModifierGroups[4].Initialize(true, 7,
					&MRVADD_SecOp_MRG_16H,
					&MRVADD_SecOp_MRG_16L,
					&MRVADD_SecOp_MRG_8B0,
					&MRVADD_SecOp_MRG_8B2,
					&MRVADD_SecOp_ACC,
					&MRVADD_SecOp_MAX,
					&MRVADD_SecOp_MIN);
		ModifierGroups[5].Initialize(true, 1, &MRS);
	}
}IRVADD;
