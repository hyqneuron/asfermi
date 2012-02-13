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

#ifndef RulesModifierExecutionDefined
#define RulesModifierExecutionDefined

struct ModifierRuleCALNOINC;
extern ModifierRuleCALNOINC MRCALNOINC;

struct ModifierRuleBRAU;
extern ModifierRuleBRAU MRBRAU, MRBRALMT;

struct ModifierRuleNOPTRIG;
extern ModifierRuleNOPTRIG MRNOPTRIG;

struct ModifierRuleNOPOP;
extern ModifierRuleNOPOP MRNOPFMA64,
						 MRNOPFMA32,
						 MRNOPXLU  ,
						 MRNOPALU  ,
						 MRNOPAGU  ,
						 MRNOPSU   ,
						 MRNOPFU   ,
						 MRNOPFMUL ;

struct ModifierRuleMEMBAR;
extern ModifierRuleMEMBAR MRMEMBARCTA, MRMEMBARGL, MRMEMBARSYS;

struct ModifierRuleATOM;
extern ModifierRuleATOM
						MRATOMADD,	
						MRATOMMIN,
						MRATOMMAX,
						MRATOMDEC,
						MRATOMINC,
						MRATOMAND,
						MRATOMOR,
						MRATOMXOR,
						MRATOMEXCH,
						MRATOMCAS;


struct ModifierRuleATOMType;
extern ModifierRuleATOMType 
						MRATOMTypeU64,
						MRATOMTypeS32,
						MRATOMTypeF32;

struct ModifierRuleATOMIgnored;
extern ModifierRuleATOMIgnored MRATOMIgnoredFTZ, MRATOMIgnoredRN;

struct ModifierRuleVOTE;
extern ModifierRuleVOTE MRVOTEALL, MRVOTEANY, MRVOTEEQ, MRVOTEVTG;

struct ModifierRuleVOTEVTG;
extern ModifierRuleVOTEVTG MRVOTEVTGR, MRVOTEVTGA, MRVOTEVTGRA;

#endif
