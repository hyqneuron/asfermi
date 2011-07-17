#ifndef RulesModifierCommonDefined


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


#else
#define RulesModifierCommonDefined
#endif