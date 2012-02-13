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

void inline ApplyModifierRuleUnconditional(ModifierRule* rule)
{
	if(rule->Apply0)
	{
		csCurrentInstruction.OpcodeWord0 &= rule->Mask0;
		csCurrentInstruction.OpcodeWord0 |= rule->Bits0;
	}
	if(csCurrentInstruction.Is8 && rule->Apply1)
	{
		csCurrentInstruction.OpcodeWord1 &= rule->Mask1;
		csCurrentInstruction.OpcodeWord1 |= rule->Bits1;
	}
}


#include "RulesModifier/RulesModifierDataMovement.h"
#include "RulesModifier/RulesModifierInteger.h"
#include "RulesModifier/RulesModifierFloat.h"
#include "RulesModifier/RulesModifierConversion.h"
#include "RulesModifier/RulesModifierCommon.h"
#include "RulesModifier/RulesModifierExecution.h"
#include "RulesModifier/RulesModifierLogic.h"
#include "RulesModifier/RulesModifierOthers.h"
