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

#ifndef SpecificParsersDefined
#define SpecificParsersDefined

#include "SubString.h"
#include "DataTypes.h"

//	1
//-----Declaration of default parsers: DefaultMasterParser, DefaultLineParser, DefaultInstructionParser, DefaultDirectiveParser
struct MasterParserDefault;
extern MasterParserDefault MPDefault;

struct LineParserDefault;
extern LineParserDefault LPDefault;

struct InstructionParserDefault;
extern InstructionParserDefault IPDefault;

struct DirectiveParserDefault;
extern DirectiveParserDefault DPDefault;
//-----End of default parser declarations


struct LineParserConstant2;
extern LineParserConstant2 LPConstant2;
void Constant2ParseInt(SubString &content);
void Constant2ParseLong(SubString &content);
void Constant2ParseFloat(SubString &content);
void Constant2ParseDouble(SubString &content);
void Constant2ParseMixed(SubString &content);

#else
#endif
