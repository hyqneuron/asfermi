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

#ifndef RulesInstructionExecutionDefined
#define RulesInstructionExecutionDefined


struct InstructionRuleEXIT;
extern InstructionRuleEXIT IREXIT;

struct InstructionRuleCAL;
extern InstructionRuleCAL IRCAL;

struct InstructionRuleJCAL;
extern InstructionRuleJCAL IRJCAL;

struct InstructionRuleSSY;
extern InstructionRuleSSY IRSSY;

struct InstructionRuleBRA;
extern InstructionRuleBRA IRBRA;

struct InstructionRuleJMP;
extern InstructionRuleJMP IRJMP;

struct InstructionRulePRET;
extern InstructionRulePRET IRPRET;

struct InstructionRuleRET;
extern InstructionRuleRET IRRET;

struct InstructionRulePBK;
extern InstructionRulePBK IRPBK;

struct InstructionRuleBRK;
extern InstructionRuleBRK IRBRK;

struct InstructionRulePCNT;
extern InstructionRulePCNT IRPCNT;

struct InstructionRuleCONT;
extern InstructionRuleCONT IRCONT;

struct InstructionRulePLONGJMP;
extern InstructionRulePLONGJMP IRPLONGJMP;

struct InstructionRuleLONGJMP;
extern InstructionRuleLONGJMP IRLONGJMP;

struct InstructionRuleNOP;
extern InstructionRuleNOP IRNOP;

struct InstructionRuleBAR;
extern InstructionRuleBAR IRBAR;

struct InstructionRuleB2R;
extern InstructionRuleB2R IRB2R;

struct InstructionRuleMEMBAR;
extern InstructionRuleMEMBAR IRMEMBAR;

struct InstructionRuleATOM;
extern InstructionRuleATOM IRATOM;

struct InstructionRuleRED;
extern InstructionRuleRED IRRED;

struct InstructionRuleVOTE;
extern InstructionRuleVOTE IRVOTE;

#else
#endif
