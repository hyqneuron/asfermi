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

#ifndef RulesOperandConstantDefined
#define RulesOperandConstantDefined




//24-bit Hexadecimal Constant Operand
struct OperandRuleImmediate24HexConstant;
extern OperandRuleImmediate24HexConstant OPRImmediate24HexConstant;


struct OperandRuleImmediate32HexConstant;
extern OperandRuleImmediate32HexConstant OPRImmediate32HexConstant;



//32-bit Integer Constant Operand
struct OperandRuleImmediate32IntConstant;
extern OperandRuleImmediate32IntConstant OPRImmediate32IntConstant;

//32-bit Floating Number Constant Operand
struct OperandRuleImmediate32FloatConstant;
extern OperandRuleImmediate32FloatConstant OPRImmediate32FloatConstant;

//32-bit Constant: Hex || Int || Float
struct OperandRuleImmediate32AnyConstant;
extern OperandRuleImmediate32AnyConstant OPRImmediate32AnyConstant;

struct OperandRuleImmediate16HexOrInt;
extern OperandRuleImmediate16HexOrInt OPRImmediate16HexOrInt, OPRImmediate16HexOrIntOptional;

struct OperandRuleS2R;
extern OperandRuleS2R OPRS2R;

#else
#endif
