
#ifndef RulesModifierLogicDefined

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

#else
#define RulesModifierLogicDefined
#endif