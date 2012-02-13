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

#ifndef RulesInstructionDataMovementDefined
#define RulesInstructionDataMovementDefined

struct InstructionRuleMOV;
extern InstructionRuleMOV IRMOV;

struct InstructionRuleMOV32I;
extern InstructionRuleMOV32I IRMOV32I;



struct InstructionRuleLD;
extern InstructionRuleLD IRLD;


struct InstructionRuleLDU;
extern InstructionRuleLDU IRLDU;


struct InstructionRuleLDL;
extern InstructionRuleLDL IRLDL;

struct InstructionRuleLDS;
extern InstructionRuleLDS IRLDS;

struct InstructionRuleLDC;
extern InstructionRuleLDC IRLDC;

struct InstructionRuleST;
extern InstructionRuleST IRST;


struct InstructionRuleSTL;
extern InstructionRuleSTL IRSTL;

struct InstructionRuleSTS;
extern InstructionRuleSTS IRSTS;

struct InstructionRuleLDLK;
extern InstructionRuleLDLK IRLDLK;

struct InstructionRuleLDSLK;
extern InstructionRuleLDSLK IRLDSLK;

struct InstructionRuleSTUL;
extern InstructionRuleSTUL IRSTUL;

struct InstructionRuleSTSUL;
extern InstructionRuleSTSUL IRSTSUL;

#else
#endif
