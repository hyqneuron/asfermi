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

/*
This file contains various helper functions used by the assembler (during the preprocess stage)

1: Main helper functions
2: Commandline stage helper functions
9: Debugging helper functions

all functions are prefixed with 'hp'
*/
#ifndef helperMixedDefined
#define helperMixedDefined

#include <fstream>

using namespace std;



//	1
//-----Main helper functions
void hpCleanUp();
//-----End of main helper functions

//	2
//----- Command-line stage helper functions
void hpUsage();
int hpHexCharToInt(char* str);
int hpFileSizeAndSetBegin(fstream &file);
int hpFindInSource(char target, int startPos, int &length);
void hpReadSource(char* path);
void hpReadSourceArray(char* source);
void hpCheckOutputForReplace(char* path, char* kernelname, char* replacepoint);
//-----End of command-line helper functions

//9
//-----Debugging functions
void hpPrintLines();
void hpPrintInstructions();
void hpPrintDirectives();	
void hpPrintComponents(Instruction &instruction);
static const char* binaryRef[16] = {"0000", "1000", "0100", "1100", "0010", "1010", "0110", "1110", 
									"0001", "1001", "0101", "1101", "0011", "1011", "0111", "1111"};
void hpPrintBinary8(unsigned int word0, unsigned int word1);
//-----End of debugging functions

//Convert binary string often seen on asfermi's site into an unsigned int
void hpBinaryStringToOpcode4(char* string, unsigned int &word0, int &i);
void hpBinaryStringToOpcode4(char* string, unsigned int &word0);
void hpBinaryStringToOpcode8(char* string, unsigned int &word0, unsigned int &word1);

#else
#endif
