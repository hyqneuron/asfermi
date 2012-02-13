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

#ifndef RulesModifierConversionDefined
#define RulesModifierConversionDefined

struct ModifierRuleF2IDest;
extern ModifierRuleF2IDest
	MRF2IDestU8,
	MRF2IDestU16,
	MRF2IDestU32,
	MRF2IDestU64,
	MRF2IDestS8,
	MRF2IDestS16,
	MRF2IDestS32,
	MRF2IDestS64; 

struct ModifierRuleF2ISource;
extern ModifierRuleF2ISource MRF2ISourceF16,MRF2ISourceF32,MRF2ISourceF64;

struct ModifierRuleF2IRound;
extern ModifierRuleF2IRound MRF2IFLOOR, MRF2ICEIL, MRF2ITRUNC,
		MRF2FRM, MRF2FRP, MRF2FRZ;

struct ModifierRuleF2IFTZ;
extern ModifierRuleF2IFTZ MRF2IFTZ;

struct ModifierRuleF2FPASS;
extern ModifierRuleF2FPASS MRF2FPASS, MRF2FROUND;

struct ModifierRuleI2FSource;
extern ModifierRuleI2FSource 
	MRI2FSourceU8,
	MRI2FSourceU16,
	MRI2FSourceU32,
	MRI2FSourceU64,
	MRI2FSourceS8,
	MRI2FSourceS16,
	MRI2FSourceS32,
	MRI2FSourceS64;

struct ModifierRuleI2FDest;
extern ModifierRuleI2FDest MRI2FDestF16,MRI2FDestF32,MRI2FDestF64;

struct ModifierRuleI2FRound;
extern ModifierRuleI2FRound MRI2FRM, MRI2FRP, MRI2FRZ;

#else
#endif
