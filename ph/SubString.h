#ifndef SubStringDefined //prevent multiple inclusion
#define SubStringDefined

 
struct SubString
{
	int Length; 
	char* Start;
	SubString(){}
	SubString(int offset, int length);
	SubString(char* target);
	char operator [] (int position);
	int Find(char target, int startPos);
	int FindBlank(int startPos);
	SubString SubStr(int startPos, int length);
	void RemoveBlankAtBeginning();
	bool Compare(SubString subString);
	bool CompareWithCharArray(char* target, int length);
	char* ToCharArray();
	void SubEndWithNull();
	void RecoverEndWithNull();	
	unsigned int ToImmediate32FromHexConstant(bool acceptNegative); 
	unsigned int ToImmediate32FromFloat32();
	unsigned int ToImmediate32FromFloat64();
	unsigned int ToImmediate32FromInt32();
	unsigned int ToImmediate32FromInt64();
	unsigned int ToImmediate32FromIntConstant(); 
	unsigned int ToImmediate32FromFloatConstant();
	void ToGlobalMemory(int &register1, unsigned int&memory);
	void ToConstantMemory(unsigned int &bank, int &register1, unsigned int &memory);
	int ToRegister();	
	unsigned int ToImmediate20FromHexConstant(bool acceptNegative);
	unsigned int ToImmediate20FromIntConstant();
	unsigned int ToImmediate20FromFloatConstant();
	
	char* ToCharArrayStopOnCR();	
};
#else
#endif