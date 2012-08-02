/*
 * Copyright (c) 2011, 2012 by Hou Yunqing
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef SubStringDefined //prevent multiple inclusion
#define SubStringDefined
#include <list>
using namespace std;
 
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
	void RemoveBlankAtEnd();
	bool Compare(SubString subString);
	bool CompareIgnoreEndingBlank(SubString subString);
	bool CompareWithCharArray(char* target, int length);
	char* ToCharArray();
	void SubEndWithNull();
	void RecoverEndWithNull();	
	unsigned int ToImmediate32FromHexConstant(bool acceptNegative); 
	unsigned int ToImmediate32FromFloat32();
	unsigned int ToImmediate32FromFloat64();
	unsigned int ToImmediate32FromInt32(bool warnZero = true);
	unsigned int ToImmediate32FromInt64();
	unsigned int ToImmediate32FromIntConstant(); 
	unsigned int ToImmediate32FromFloatConstant();
	void ToGlobalMemory(int &register1, unsigned int&memory);
	void ToConstantMemory(unsigned int &bank, int &register1, unsigned int &memory, int maxBank = 15);
	int ToRegister();	
	unsigned int ToImmediate20FromHexConstant(bool acceptNegative);
	unsigned int ToImmediate20FromIntConstant();
	unsigned int ToImmediate20FromFloatConstant();
	inline bool IsRegister()
	{
		return Start[0]=='R'||Start[0]=='r';
	}
	inline bool IsConstantMemory()
	{
		return Start[0]=='c' || Start[0]=='C';
	}
	inline bool IsHex()
	{
		return Length>2&& Start[0]=='0' &&(Start[1]=='x'||Start[1]=='X');
	}
	inline bool IsFloat()
	{
		return Start[0]=='F';
	}
	
	char* ToCharArrayStopOnCR();	
};


struct SortElement
{
	void *ExtraInfo;
	SubString Name;

	SortElement(){}
	SortElement(void *extraInfo, SubString name);
};
extern SortElement SortNotFound;
unsigned int SortComputeIndex(SubString subString);
void SortInitialize(list<SortElement>elementList, SortElement* &sortedList, unsigned int* &indicesList);
SortElement SortFind(SortElement* sortedList, unsigned int* indicesList, unsigned int count, SubString target);

#endif
