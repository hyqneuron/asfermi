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

#ifndef RulesOperandRegisterDefined
#define RulesOperandRegisterDefined


struct OperandRuleRegister;
extern OperandRuleRegister OPRRegister0, OPRRegister1, OPRRegister2;

//reg3 used a separate rule because it applies it result to OpcodeWord1 instead of 0
struct OperandRuleRegister3;
extern OperandRuleRegister3 OPRRegister3ForMAD, OPRRegister3ForCMP, OPRRegister3ForATOM, OPRRegister4ForATOM;

//Note that some operands can have modifiers
//This rule deals with registers that can have the .CC modifier
struct OperandRuleRegisterWithCC;
extern OperandRuleRegisterWithCC OPRRegisterWithCC4IADD32I, OPRRegisterWithCCAt16;

struct OperandRuleRegister0ForMemory;
extern OperandRuleRegister0ForMemory OPRRegister0ForMemory;

//Predicate register operand
struct OperandRulePredicate;
extern OperandRulePredicate OPRPredicate1, 
							OPRPredicate0,
							OPRPredicate2NotNegatable,
							OPRPredicateForLDSLK, 
							OPRPredicateForBAR,
							OPRPredicate0ForVOTE,
							OPRPredicate1ForVOTENoteNegatable;

//Some predicate registers expressions can be negated with !
//this kind of operand is processed separately
struct OperandRulePredicate2;
extern OperandRulePredicate2 OPRPredicate2, OPRPredicate1ForVOTE, OPRPredicate3ForPSETP;

struct OperandRulePredicateForLDLK;
extern OperandRulePredicateForLDLK OPRPredicateForLDLK;


struct OperandRuleFADD32IReg1;
extern OperandRuleFADD32IReg1 OPRFADD32IReg1;

struct OperandRuleRegister1WithSignFlag;
extern OperandRuleRegister1WithSignFlag OPRIMADReg1, OPRISCADDReg1;



//Register&composite operands for D***
struct OperandRuleRegister0ForDouble;
extern OperandRuleRegister0ForDouble OPRRegister0ForDouble;

struct OperandRuleRegister1ForDouble;
extern OperandRuleRegister1ForDouble OPRRegister1ForDoubleWith2OP, OPRRegister1ForDouble;

struct OperandRuleCompositeOperandForDouble;
extern OperandRuleCompositeOperandForDouble OPRCompositeForDoubleWith2OP, OPRCompositeForDoubleWith1OP;

struct OperandRuleRegister3ForDouble;
extern OperandRuleRegister3ForDouble OPRRegister3ForDouble;



#endif
