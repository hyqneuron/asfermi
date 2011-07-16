/*
This file contains rules for modifiers
*/
#ifndef RulesModifierDefined

//-----Specific modifier rules

struct ModifierRuleLDType: ModifierRule
{
	ModifierRuleLDType(int type): ModifierRule("", true, false, false)
	{		
		::InstructionRule::BinaryStringToOpcode4("11111000111111111111111111111111", Mask0);
		Bits0 = type<<5;
		switch(type)
		{
		case 0:
			Name = "U8";
			break;
		case 1:
			Name = "S8";
			break;
		case 2:
			Name = "U16";
			break;
		case 3:
			Name = "S16";
			break;
		case 5:
			Name = "64";
			break;
		case 6:
			Name = "128";
			break;
		default:
			throw exception("Unsupported modifier");
			break;
		}
	}
}MRLDU8(0), MRLDS8(1), MRLDU16(2), MRLDS16(3), MRLD64(5), MRLD128(6);

struct ModifierRuleLDCop: ModifierRule
{
	ModifierRuleLDCop(int type): ModifierRule("", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111001111111111111111111111", Mask0);
		Bits0 = type << 8;
		if(type==1)
			Name = "CG";
		else if(type==2)
			Name = "CS";
		else if(type == 3)
			Name = "CV";
		else if(type == 5)
		{
			Name = "LU";
			Bits0 = 2<<8;
		}
	}
}MRLDCopCG(1),MRLDCopCS(2),MRLDCopCV(3), MRLDCopLU(5);


struct ModifierRuleSTCop: ModifierRule
{
	ModifierRuleSTCop(int type): ModifierRule("", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111001111111111111111111111", Mask0);
		Bits0 = type << 8;
		if(type==1)
			Name = "CG";
		else if(type==2)
			Name = "CS";
		else if(type == 3)
			Name = "WT";
	}
}MRSTCopCG(1),MRSTCopCS(2),MRSTCopWT(3);

struct ModifierRuleSETPLogic: ModifierRule
{
	ModifierRuleSETPLogic(int type) : ModifierRule("", false, true, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111111111100111111111", Mask1);
		//AND
		if(type==0)
		{
		Name = "AND";
		::InstructionRule::BinaryStringToOpcode4("00000000000000000000000000000000", Bits1);
		}
		//OR
		else if(type==1)
		{
		Name = "OR";
		::InstructionRule::BinaryStringToOpcode4("00000000000000000000010000000000", Bits1);
		}
		//XOR
		else
		{
		Name = "XOR";
		::InstructionRule::BinaryStringToOpcode4("00000000000000000000001000000000", Bits1);
		}
	}
}MRSETPLogicAND(0), MRSETPLogicOR(1), MRSETPLogicXOR(2);

struct ModifierRuleSETPComparison: ModifierRule
{
	ModifierRuleSETPComparison(int type): ModifierRule("", false, true, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111111111111000011111", Mask1);
		Bits1 = type<<23;
		
		switch(type)
		{
		case 1: //LT
			Name = "LT";
			break;
		case 2: //EQ
			Name = "EQ";
			break;
		case 3: //LE
			Name = "LE";
			break;
		case 4: //GT
			Name = "GT";
			break;
		case 5: //NE
			Name = "NE";
			break;
		case 6: //GE
			Name = "GE";
			break;
		case 7: //NUM
			Name = "NUM";
			break;
		case 8: //NAN
			Name = "NAN";
			break;
		case 9: //LTU
			Name = "LTU";
			break;
		case 10://EQU
			Name = "EQU";
			break;
		case 11://LEU
			Name = "LEU";
			break;
		case 12://GTU
			Name = "GTU";
			break;
		case 13://NEU
			Name = "NEU";
			break;
		case 14://GEU
			Name = "GEU";
			break;
		default:
			throw exception("Unknown SETP comparison modifier");
		};
	}
}MRSETPComparisonLT(1),
	MRSETPComparisonEQ(2),
	MRSETPComparisonLE(3),
	MRSETPComparisonGT(4),
	MRSETPComparisonNE(5),
	MRSETPComparisonGE(6),
	MRSETPComparisonNUM(7),
	MRSETPComparisonNAN(8),
	MRSETPComparisonLTU(9),
	MRSETPComparisonEQU(10),
	MRSETPComparisonLEU(11),
	MRSETPComparisonGTU(12),
	MRSETPComparisonNEU(13),
	MRSETPComparisonGEU(14);
struct ModifierRuleIMUL0U32: ModifierRule
{
	ModifierRuleIMUL0U32(): ModifierRule("U32", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111010111111111111111111111111", Mask0);
		Bits0 = 0;
	}
}MRIMUL0U32;

struct ModifierRuleIMUL1U32: ModifierRule
{
	ModifierRuleIMUL1U32(): ModifierRule("U32", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111011111111111111111111111111", Mask0);
		Bits0 = 0;
	}
}MRIMUL1U32;


struct ModifierRuleIMUL0S32: ModifierRule
{
	ModifierRuleIMUL0S32(): ModifierRule("S32", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111010111111111111111111111111", Mask0);
		::InstructionRule::BinaryStringToOpcode4("00000101000000000000000000000000", Bits0);
	}
}MRIMUL0S32;

struct ModifierRuleIMUL1S32: ModifierRule
{
	ModifierRuleIMUL1S32(): ModifierRule("S32", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111011111111111111111111111111", Mask0);
		::InstructionRule::BinaryStringToOpcode4("00000100000000000000000000000000", Bits0);
	}
}MRIMUL1S32;

struct ModifierRuleIMULHI: ModifierRule
{
	ModifierRuleIMULHI(): ModifierRule("HI", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111101111111111111111111111111", Mask0);
		::InstructionRule::BinaryStringToOpcode4("00000010000000000000000000000000", Bits0);
	}
}MRIMULHI;

struct ModifierRuleIMULSAT: ModifierRule
{
	ModifierRuleIMULSAT(): ModifierRule("SAT", false, true, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111111111111101111111", Mask1);
		::InstructionRule::BinaryStringToOpcode4("00000000000000000000000010000000", Bits1);
	}
}MRIMULSAT;

struct ModifierRuleFMULR: ModifierRule
{
	ModifierRuleFMULR(int type, char* name): ModifierRule("", false, true, false)
	{
		Name = name;
		::InstructionRule::BinaryStringToOpcode4("11111111111111111111111001111111", Mask1);
		Bits1 = type<<23;
	}
}MRFMULRM(1, "RM"), MRFMULRP(2, "RP"), MRFMULRZ(3, "RZ");



struct ModifierRuleFMULSAT: ModifierRule
{
	ModifierRuleFMULSAT(): ModifierRule("SAT", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111011111111111111111111111111", Mask0);
		Bits0 = 1<<5;
	}
}MRFMULSAT;

struct ModifierRuleCALNOINC: ModifierRule
{
	ModifierRuleCALNOINC(): ModifierRule("NOINC", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111110111111111111111", Mask0);
		Bits0 = 0;
	}
}MRCALNOINC;

struct ModifierRuleBRAU: ModifierRule
{
	ModifierRuleBRAU(): ModifierRule("U", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111111111111101111111111111111", Mask0);
		Bits0 = 1<<15;
	}
}MRBRAU;

struct ModifierRuleLOP: ModifierRule
{
	ModifierRuleLOP(int type): ModifierRule("", true, false, false)
	{
		::InstructionRule::BinaryStringToOpcode4("11111100111111111111111111111111", Mask0);
		Bits0 = type << 6;
		if(type==0)
			Name = "AND";
		else if(type==1)
			Name = "OR";
		else if(type==2)
			Name = "XOR";
		else if(type ==3)
			Name = "PASS_B";
	}
}MRLOPAND(0), MRLOPOR(1), MRLOPXOR(2), MRLOPPASS(3);

//-----End of specific modifier rules

#else
#define RulesModifierDefined
#endif