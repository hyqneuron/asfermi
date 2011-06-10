#if defined SubStringDefined //prevent multiple inclusion
#else
#define SubStringDefined yes
//-----Start of code
#include <vld.h>


using namespace std;

extern char* csSource;

int d_currentPos; //used by various functions in SubString
static const unsigned int op_HexRef[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
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
	int FindInSource(char target, int startPos, int &lengthFromStartPos) //startPos is the position in this string. lengthFromStart is the d_currentPos - startPos
	{
		for(d_currentPos = startPos; d_currentPos < Length; d_currentPos++)
		{
			if(target == Start[d_currentPos])
			{
				lengthFromStartPos = d_currentPos - startPos;
				return d_currentPos + Offset;
			}
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
	bool CompareWithCharArrayIgnoreEndingBlank(char* target, int length) //length is the length of the char
	{
		if(length < Length)
			return false;
		for(int i =0; i<length; i++)
		{
			if(Start[i]!=target[i])
				return false;
		}
		for(int i = Length; Length < length; i++) //it's fine for the target to end with blanks
		{
			if(target[i]>32)
				return false;
		}
		return true;
	}
	char* ToCharArrayStopOnCR();
	int ToRegister();
	unsigned int ToImmediate32FromHex();
	void ToGlobalMemory(int &register1, unsigned int&memory);
	
};

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
int SubString:: ToRegister()
	{
		int result;
		if(Length<2 || Start[0]!='R')
		{
			throw 104; //Incorrect register
			//return;
		}
		if(Start[1] == 'Z')
		{
			return 63;
		}
		result = (int)Start[1] - 48;
		if(result<0 || result>9)
		{
			throw 104; //incorrect register
			//return;
		}
		if(Length==2)
		{
			return result;
		}
		int n2 = (int)Start[2] - 48;
		if(n2<0 || n2>9)
		{
			return result;
		}
		result *= 10;
		result += n2;
		if(result>=63)
		{
			throw 105; //register number too large
			//return;
		}
		return result;
	}

#define mReduceWithError(o,error)				\
{												\
	if(o<58)									\
	{											\
		if(o<48)								\
		{										\
			result >>= shift + 4;				\
			break;								\
		}										\
		o-=48;									\
	}											\
	else if(o<71)								\
	{											\
		if(o<65)								\
		{										\
			result >>= shift + 4;				\
			break;								\
		}										\
		o-=55;									\
	}											\
	else if(o<103)								\
	{											\
		if(o<97)								\
		{										\
			result >>= shift + 4;				\
			break;								\
		}										\
		o -= 87;								\
	}											\
	else										\
	{											\
		result >>= shift + 4;					\
		break;									\
	}											\
}
	
unsigned int SubString::ToImmediate32FromHex()
	{
		if(Length<3 || Start[0]!='0' || (Start[1]!='x'&& Start[1] != 'X') )
		{
			throw 106; //incorrect hex
			//return;
		}
		unsigned int  result = 0;
		int maxlength = (Length<10)? Length:10;
		int shift = Length * 4 -12;
		int digit;
		int i;
		for(i =2; i<maxlength; i++)
		{
			digit = (int) Start[i];
			mReduceWithError(digit, 58); //58: invalid hex constant
			result |= op_HexRef[digit] << shift;
			shift -= 4;
		}
		if(i==2)
		{
			throw 106;
			//return;
		}
		return result;
	}
void SubString::ToGlobalMemory(int &register1, unsigned int&memory)
	{
		register1 = 63; //RZ
		memory = 0;
		if(Length < 4 || Start[0]!='[') //[R0]
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
			if(Start[1]=='R')
				register1 = (SubStr(startPos, Length -startPos)).ToRegister();
			else
				memory = SubStr(startPos, Length -startPos ).ToImmediate32FromHex();
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
			memory = SubStr(startPos, Length - startPos).ToImmediate32FromHex();
		}
	}


#endif