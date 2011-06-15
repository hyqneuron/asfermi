#if defined SubStringDefined //prevent multiple inclusion
#else
#define SubStringDefined yes
//-----Start of code
//#include <vld.h> //remove when you compile
#include <math.h>


using namespace std;

extern char* csSource;
extern void hpWarning(int e);

int d_currentPos; //used by various functions in SubString

struct SubString
{
	int Offset;	//global offset in csSource
	int Length;
	char* Start;
	SubString(){}
	SubString(int offset, int length)
	{
		Offset = offset;
		Length = length;
		Start = csSource + offset;
		if(length<0)
			throw exception();
	}
	char operator [] (int position)
	{
		return Start[position];
	}
	int Find(char target, int startPos) //startPos is the position in this string
	{
		for(d_currentPos = startPos; d_currentPos < Length; d_currentPos++)
		{
			if(target == Start[d_currentPos])
				return d_currentPos;
		}
		return -1;
	}
	int FindBlank(int startPos)
	{
		for(d_currentPos = startPos; d_currentPos < Length; d_currentPos++)
		{
			if(Start[d_currentPos] < 33)
				return d_currentPos;
		}
		return -1;
	}
	SubString SubStr(int startPos, int length)
	{
		SubString result(startPos + Offset, length);
		if(length<0)
			throw exception();
		return result;
	}
	char* ToCharArray()
	{
		char *result = new char[Length + 1];
		for(int i =0 ; i<Length; i++)
			result[i] = Start[i];
		result[Length] = (char)0;
		return result;
	}
	bool CompareWithCharArray(char* target, int length) //length is the length of the char
	{
		if(length < Length)
			return false;
		for(int i =0; i<length; i++)
		{
			if(Start[i]!=target[i])
				return false;
		}
		return true;
	}

//syntax checking are done in primary  substring functions as well as in composite operand processors
//but not in 20-bit functions
	
	unsigned int ToImmediate32FromHexConstant(); //check
	unsigned int ToImmediate32FromFloat32(int modLength); //check
	unsigned int ToImmediate32FromFloat64(int modLength); //check
	unsigned int ToImmediate32FromInt32(); //check
	unsigned int ToImmediate32FromInt64(); //check

	unsigned int ToImmediate32FromIntConstant(); //check
	unsigned int ToImmediate32FromFloatConstant(int modLength); //check

	void ToGlobalMemory(int &register1, unsigned int&memory);
	void ToConstantMemory(unsigned int &bank, int &register1, unsigned int &memory);
	int ToRegister();

	
	unsigned int ToImmediate20FromHexConstant()
	{
		unsigned int result = ToImmediate32FromHexConstant();
		if(result>0xFFFFF)
			throw 113;
		return result;
	}

	unsigned int ToImmediate20FromIntConstant()
	{
		if(Start[0] == 'H' || Start[0] == 'L')
			throw 118; //20-bit cannot contain 64-bit
		unsigned int result = ToImmediate32FromInt32();
		if(result>0xFFFFF)
			throw 113;
		return result;
	}

	unsigned int ToImmediate20FromFloatConstant(int modLength)
	{
		if(Length<2 || Start[0] != 'F') //need to check this to ensure access to Start[1] doesn't yield error
			throw 117; //Incorrect floating number
		if(Start[1] == 'H' || Start[1] == 'L')
			throw 118; //20-bit cannot contain 64-bit
		else
			return ToImmediate32FromFloat32(modLength) >> 12; //issue: no warning regarding loss of precision
	}
	
	char* ToCharArrayStopOnCR();	
};





static const unsigned int op_HexRef[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
unsigned int SubString::ToImmediate32FromHexConstant()
{
	if(Length<3 || Start[0]!='0' || (Start[1]!='x'&& Start[1] != 'X') )
		throw 106; //incorrect hex
	unsigned int  result = 0;
	int maxlength = (Length<10)? Length:10;

	int digit;
	int i;
	for(i =2; i<maxlength; i++)
	{
		digit = (int) Start[i];
		
		if(digit<58)
		{
			if(digit<48)
				break;
			digit-=48;
		}
		else if(digit<71)
		{
			if(digit<65)
				break;
			digit-=55;
		}
		else if(digit<103)
		{
			if(digit<97)
				break;
			digit -= 87;
		}
		result <<=4;
		result |= op_HexRef[digit];
	}
	if(i==2)
		throw 106;
	return result;
}
float ti32_PINF = 1000000000000000000000000000000000000000000000000000000000000000000000.0;
float ti32_NINF = -1000000000000000000000000000000000000000000000000000000000000000000000.0;
unsigned int SubString::ToImmediate32FromFloat32(int modLength)
{
	if(Length<2 || Start[0]!='F') //At least F0
		throw 121; //incorrect floating point number format
	int zeroPos = Length + modLength;
	if(modLength!=0)
		zeroPos++;
	char zeroSaver = Start[zeroPos];
	Start[zeroPos] = (char)0;

	float fresult = atof(Start+1);
	Start[zeroPos] = zeroSaver;
	if(fresult == 0.0)
		hpWarning(10); //evaluation of constant returned zero
	else if(fresult == ti32_PINF || fresult == ti32_NINF)
		hpWarning(11); //overflow
	return *(unsigned int *)&fresult;
}


unsigned int SubString::ToImmediate32FromFloat64(int modLength)
{
	if(Length<3 || Start[0]!='F' || (Start[1]!='H' && Start[1]!='L') ) //At least FH0/FL0
		throw 121; //incorrect floating point number format

	int zeroPos = Length + modLength;
	if(modLength!=0)
		zeroPos++;
	char zeroSaver = Start[zeroPos];
	Start[zeroPos] = (char)0;

	double dresult = atof(Start+2);
	Start[zeroPos] = zeroSaver;

	if(dresult == 0.0)
		hpWarning(10); //evaluation of constant returned zero. Issue: suggestion of using RZ maybe incorrect.
	else if(dresult == HUGE_VAL || dresult == -HUGE_VAL) //defined in math.h
		hpWarning(11);//overflow
	unsigned int *resultbits = (unsigned int *)&dresult;
	if(Start[1]=='H')
		resultbits++;
	return *resultbits;
}
unsigned int SubString::ToImmediate32FromInt32()
{
	char zeroSaver = Start[Length];
	Start[Length] = 0;
	int result = atoi(Start);
	Start[Length] = zeroSaver;

	if(result == 0)
		hpWarning(10);
	else if(result == INT_MAX || result == INT_MIN)
		hpWarning(11);
	return *(unsigned int*)&result; //issue: can't deal with large unsigned integers
}
unsigned int SubString::ToImmediate32FromInt64()
{
	char zeroSaver = Start[Length];
	Start[Length] = 0;
	long result = atol(Start+1); //first character is H or L
	Start[Length] = zeroSaver;

	if(result == 0)
		hpWarning(10);
	else if(result == LONG_MAX || result == LONG_MIN)
		hpWarning(11);

	unsigned int *resultbits = (unsigned int *)&result;
	if(Start[0]=='H')
		resultbits++;
	return *resultbits;
}

unsigned int SubString::ToImmediate32FromIntConstant()
{
	if(Start[0]=='H' || Start[0]=='L') //long
	{
		return ToImmediate32FromInt64();
	}
	else //Int
	{
		return ToImmediate32FromInt32();
	}
}
unsigned int SubString::ToImmediate32FromFloatConstant(int modLength)
{
	if(Length<2) //F0, ToImmediate32FromDouble assumes length 2 or above
		throw 117; //Incorrect floating number
	if(Start[1] == 'H' || Start[1] == 'L')
		return ToImmediate32FromFloat64(modLength);
	else
		return ToImmediate32FromFloat32(modLength);
}


int SubString:: ToRegister()
{
	int result;
	if(Length<2 || Start[0]!='R')
		throw 104; //Incorrect register
	if(Start[1] == 'Z')
		return 63;
	result = (int)Start[1] - 48;
	if(result<0 || result>9)
		throw 104; //incorrect register
	if(Length==2)
		return result;
	int n2 = (int)Start[2] - 48;
	if(n2<0 || n2>9)
		return result;
	result *= 10;
	result += n2;
	if(result>=63)
		throw 105; //register number too large
	return result;
}


void SubString::ToGlobalMemory(int &register1, unsigned int&memory)
{
	register1 = 63; //RZ
	memory = 0;
	if(Length < 3 || Start[0]!='[') //[0]
	{
		throw 107; //incorrect global mem
		//return;
	}
	int startPos = 1;
	while(startPos<Length)
	{
		if(Start[startPos] > 32)
			break;
		startPos++;
	}
	int plusPos = Find('+', 0);
	if(plusPos==-1)
	{
		if(Start[startPos]=='R')
			register1 = (SubStr(startPos, Length -startPos)).ToRegister();
		else //Issue: what about integer values?
			memory = SubStr(startPos, Length -startPos ).ToImmediate32FromHexConstant();
	}
	else
	{
		register1 = SubStr(startPos, Length -startPos).ToRegister();
		startPos = plusPos+1;
		while(startPos<Length)
		{
			if(Start[startPos] > 32)
				break;
			startPos++;
		}
		memory = SubStr(startPos, Length - startPos).ToImmediate32FromHexConstant();
	}
}
void SubString::ToConstantMemory(unsigned int &bank, int &register1, unsigned int &memory)
{
	if(Length<11|| Start[0]!='c') //c[0][0]
		throw 110; //incorrect constant memory format
	int startPos;
	for(startPos = 1; startPos<Length; startPos++)
	{
		if(Start[startPos]>32)
			break;
	}
	if(startPos==Length || Start[startPos]!='[')
		throw 110;
	startPos++;
	int endPos = Find(']', startPos);
	if(endPos == -1)
		throw 110;
	bank = SubStr(startPos, endPos - startPos).ToImmediate32FromHexConstant(); //issue: the error line would be for global mem
	if(bank > 10)
		throw 114; //bank number too large
	startPos = endPos + 1;
	for(; startPos<Length; startPos++)
	{
		if(Start[startPos]>32)
			break;
	}
	if(startPos>=Length || Start[startPos]!='[')
		throw 110;
	SubStr(startPos, Length - startPos).ToGlobalMemory(register1, memory);
	if(memory>0xFFFF) 
		throw 111; //too large memory address
}





//-----Miscellaneous functions for SubString
char* SubString::ToCharArrayStopOnCR()
{
		char *result = new char[Length + 1];
		for(int i =0 ; i<Length; i++)
		{
			if((int)Start[i]==13) //carriage return
			{
				result[i] = (char)0;
				return result;
			}
			result[i] = Start[i];
		}
		result[Length] = (char)0;
		return result;
}

#endif