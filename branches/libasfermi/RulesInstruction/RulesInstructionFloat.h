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

#ifndef RulesInstructionFloatDefined
#define RulesInstructionFloatDefined


struct InstructionRuleFADD;
extern InstructionRuleFADD IRFADD;

struct InstructionRuleFADD32I;
extern InstructionRuleFADD32I IRFADD32I;

struct InstructionRuleFMUL;
extern InstructionRuleFMUL IRFMUL;

struct InstructionRuleFMUL32I;
extern InstructionRuleFMUL32I IRFMUL32I;

struct InstructionRuleFFMA;
extern InstructionRuleFFMA IRFFMA;

struct InstructionRuleFSETP;
extern InstructionRuleFSETP IRFSETP;

struct InstructionRuleFCMP;
extern InstructionRuleFCMP IRFCMP;

struct InstructionRuleMUFU;
extern InstructionRuleMUFU IRMUFU;

struct InstructionRuleDADD;
extern InstructionRuleDADD IRDADD;

struct InstructionRuleDMUL;
extern InstructionRuleDMUL IRDMUL;

struct InstructionRuleDFMA;
extern InstructionRuleDFMA IRDFMA;

struct InstructionRuleDSETP;
extern InstructionRuleDSETP IRDSETP;

#endif
