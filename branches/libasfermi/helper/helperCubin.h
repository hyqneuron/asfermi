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
This file contains various helper functions used during cubin output (the post-processin stage)
1: Stage1 functions
2: Stage2 functions
3: Stage3 functions
4: Stage4 and later functions

all functions are prefixed with 'hpCubin'
*/

#ifndef helperCubinDefined //prevent multiple inclusion
#define helperCubinDefined
//---code starts ---
#include "../Cubin.h"

void hpCubinSet64(bool is64);

//	1
//-----Stage1 functions
//SectionIndex, SectionSize, OffsetFromFirst, SHStrTabOffset 
void hpCubinStage1SetSection(ELFSection &section, SectionType nvType, unsigned int kernelNameLength);
void hpCubinStage1();

//	2
//-----Stage2 functions
inline void hpCubinAddSectionName1(unsigned char* sectionContent, int &offset, char* sectionPrefix, SubString &kernelName);
inline void hpCubinAddSectionName2(unsigned char* sectionContent, int &offset, const char* sectionName);
inline void hpCubinAddSectionName3(unsigned char* sectionContent, int &offset, SubString &kernelName);
inline void hpCubinStage2SetSHStrTabSectionContent();
inline void hpCubinStage2SetStrTabSectionContent();
inline void hpCubinStage2AddSectionSymbol(ELFSection &section, ELFSymbolEntry &entry, int &offset, unsigned int size);
inline void hpCubinStage2SetSymTabSectionContent();
void hpCubinStage2();

//	3
//-----Stage3
void hpCubinStage3();

//	4
//-----Stage4 and later functions
void hpCubinSetELFSectionHeader1(ELFSection &section, unsigned int type, unsigned int alignment, unsigned int &offset);

//Stage4: Setup all section headers
void hpCubinStage4();

//Stage5: Setup all program segments
void hpCubinStage5();

//Stage6: Setup ELF header
void hpCubinStage6();

//Stage7: Write to cubin
void hpCubinStage7(std::iostream& csOutput);
//-----End of cubin helper functions
#else
#endif
