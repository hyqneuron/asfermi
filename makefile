VPATH=	helper\
	RulesModifier\
	RulesOperand\
	RulesInstruction

main =	libasfermi\
	Cubin\
	DataTypes\
	GlobalVariables\
	RulesDirective\
	SpecificParsers\
	SubString

exe =	asfermi.exe.o\

lib =	asfermi.lib.o\

helper= helperCubin\
	helperException\
	helperMixed\
	helperParse

instructions = \
	RulesInstructionConversion\
	RulesInstructionDataMovement\
	RulesInstructionExecution\
	RulesInstructionFloat\
	RulesInstructionInteger\
	RulesInstructionLogic\
	RulesInstructionMiscellaneous

operands = \
	RulesOperandComposite\
	RulesOperandConstant\
	RulesOperandMemory\
	RulesOperandOthers\
	RulesOperandRegister

modifiers = \
	RulesModifierCommon\
	RulesModifierConversion\
	RulesModifierDataMovement\
	RulesModifierExecution\
	RulesModifierFloat\
	RulesModifierInteger\
	RulesModifierLogic\
	RulesModifierOthers

allnames = $(main) $(helper) $(instructions) $(operands) $(modifiers)
allobjects = $(allnames:%=%.o)

ifdef ProgramFiles
	RM = del
	CP = copy
else
	RM = rm
	CP = cp
endif

CXX = g++
CXXFLAGS = -g -w -fPIC -std=c++0x

all: asfermi libasfermi.so

asfermi: $(exe) $(allobjects)
	g++ $(exe) $(allobjects) -o $@

libasfermi.so: $(lib) $(allobjects)
	g++ $(lib) $(allobjects) -shared -o $@

clean:
	$(RM) -rf *.o
	$(RM) -rf asfermi libasfermi.so

asfermi.exe.o: asfermi.cpp SubString.h DataTypes.h Cubin.h GlobalVariables.h \
 helper.h helper/helperMixed.h helper/helperParse.h helper/helperCubin.h \
 helper/../Cubin.h helper/helperException.h SpecificParsers.h stdafx.h \
 RulesModifier.h RulesModifier/RulesModifierDataMovement.h \
 RulesModifier/RulesModifierInteger.h RulesModifier/RulesModifierFloat.h \
 RulesModifier/RulesModifierConversion.h \
 RulesModifier/RulesModifierCommon.h \
 RulesModifier/RulesModifierExecution.h \
 RulesModifier/RulesModifierLogic.h RulesModifier/RulesModifierOthers.h \
 RulesOperand.h RulesOperand/RulesOperandConstant.h \
 RulesOperand/RulesOperandRegister.h RulesOperand/RulesOperandMemory.h \
 RulesOperand/RulesOperandComposite.h RulesOperand/RulesOperandOthers.h \
 RulesInstruction.h RulesInstruction/RulesInstructionDataMovement.h \
 RulesInstruction/RulesInstructionExecution.h \
 RulesInstruction/RulesInstructionFloat.h \
 RulesInstruction/RulesInstructionInteger.h \
 RulesInstruction/RulesInstructionConversion.h \
 RulesInstruction/RulesInstructionMiscellaneous.h \
 RulesInstruction/RulesInstructionLogic.h RulesDirective.h
	g++ $(CXXFLAGS) -c $< -o $@
asfermi.lib.o: asfermi.cpp SubString.h DataTypes.h Cubin.h GlobalVariables.h \
 helper.h helper/helperMixed.h helper/helperParse.h helper/helperCubin.h \
 helper/../Cubin.h helper/helperException.h SpecificParsers.h stdafx.h \
 RulesModifier.h RulesModifier/RulesModifierDataMovement.h \
 RulesModifier/RulesModifierInteger.h RulesModifier/RulesModifierFloat.h \
 RulesModifier/RulesModifierConversion.h \
 RulesModifier/RulesModifierCommon.h \
 RulesModifier/RulesModifierExecution.h \
 RulesModifier/RulesModifierLogic.h RulesModifier/RulesModifierOthers.h \
 RulesOperand.h RulesOperand/RulesOperandConstant.h \
 RulesOperand/RulesOperandRegister.h RulesOperand/RulesOperandMemory.h \
 RulesOperand/RulesOperandComposite.h RulesOperand/RulesOperandOthers.h \
 RulesInstruction.h RulesInstruction/RulesInstructionDataMovement.h \
 RulesInstruction/RulesInstructionExecution.h \
 RulesInstruction/RulesInstructionFloat.h \
 RulesInstruction/RulesInstructionInteger.h \
 RulesInstruction/RulesInstructionConversion.h \
 RulesInstruction/RulesInstructionMiscellaneous.h \
 RulesInstruction/RulesInstructionLogic.h RulesDirective.h
	g++ $(CXXFLAGS) -DNO_MAIN -c $< -o $@
Cubin.o: Cubin.cpp Cubin.h SubString.h DataTypes.h stdafx.h \
 GlobalVariables.h helper.h helper/helperMixed.h helper/helperParse.h \
 helper/helperCubin.h helper/../Cubin.h helper/helperException.h \
 SpecificParsers.h
DataTypes.o: DataTypes.cpp DataTypes.h SubString.h stdafx.h Cubin.h \
 GlobalVariables.h helper.h helper/helperMixed.h helper/helperParse.h \
 helper/helperCubin.h helper/../Cubin.h helper/helperException.h \
 SpecificParsers.h
GlobalVariables.o: GlobalVariables.cpp GlobalVariables.h DataTypes.h \
 SubString.h Cubin.h stdafx.h helper.h helper/helperMixed.h \
 helper/helperParse.h helper/helperCubin.h helper/../Cubin.h \
 helper/helperException.h SpecificParsers.h
RulesDirective.o: RulesDirective.cpp DataTypes.h SubString.h \
 GlobalVariables.h Cubin.h helper/helperException.h SpecificParsers.h \
 stdafx.h helper.h helper/helperMixed.h helper/helperParse.h \
 helper/helperCubin.h helper/../Cubin.h RulesDirective.h \
 RulesOperand/RulesOperandComposite.h
SpecificParsers.o: SpecificParsers.cpp SpecificParsers.h SubString.h \
 DataTypes.h GlobalVariables.h Cubin.h helper/helperException.h \
 helper/helperParse.h stdafx.h helper.h helper/helperMixed.h \
 helper/helperCubin.h helper/../Cubin.h
SubString.o: SubString.cpp SubString.h stdafx.h Cubin.h DataTypes.h \
 GlobalVariables.h helper.h helper/helperMixed.h helper/helperParse.h \
 helper/helperCubin.h helper/../Cubin.h helper/helperException.h \
 SpecificParsers.h
helperCubin.o: helper/helperCubin.cpp helper/helperCubin.h \
 helper/../Cubin.h helper/../SubString.h helper/../DataTypes.h \
 helper/../GlobalVariables.h helper/../Cubin.h helper/../DataTypes.h \
 helper/../stdafx.h helper/../GlobalVariables.h helper/../helper.h \
 helper/../helper/helperMixed.h helper/../helper/helperParse.h \
 helper/../helper/helperCubin.h helper/../helper/helperException.h \
 helper/../SpecificParsers.h helper/stdafx.h
helperException.o: helper/helperException.cpp helper/../DataTypes.h \
 helper/../SubString.h helper/../GlobalVariables.h helper/../DataTypes.h \
 helper/../Cubin.h helper/helperException.h helper/../stdafx.h \
 helper/../GlobalVariables.h helper/../helper.h \
 helper/../helper/helperMixed.h helper/../helper/helperParse.h \
 helper/../helper/helperCubin.h helper/../helper/../Cubin.h \
 helper/../helper/helperException.h helper/../SpecificParsers.h \
 helper/stdafx.h
helperMixed.o: helper/helperMixed.cpp helper/../GlobalVariables.h \
 helper/../DataTypes.h helper/../SubString.h helper/../Cubin.h \
 helper/helperMixed.h helper/../stdafx.h helper/../GlobalVariables.h \
 helper/../helper.h helper/../helper/helperMixed.h \
 helper/../helper/helperParse.h helper/../helper/helperCubin.h \
 helper/../helper/../Cubin.h helper/../helper/helperException.h \
 helper/../SpecificParsers.h helper/stdafx.h
helperParse.o: helper/helperParse.cpp helper/../SubString.h \
 helper/../GlobalVariables.h helper/../DataTypes.h helper/../SubString.h \
 helper/../Cubin.h helper/helperParse.h helper/../stdafx.h \
 helper/../GlobalVariables.h helper/../helper.h \
 helper/../helper/helperMixed.h helper/../helper/helperParse.h \
 helper/../helper/helperCubin.h helper/../helper/../Cubin.h \
 helper/../helper/helperException.h helper/../SpecificParsers.h \
 helper/stdafx.h
RulesInstructionConversion.o: \
 RulesInstruction/RulesInstructionConversion.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionConversion.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesInstructionDataMovement.o: \
 RulesInstruction/RulesInstructionDataMovement.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionDataMovement.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesInstructionExecution.o: \
 RulesInstruction/RulesInstructionExecution.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionExecution.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesInstructionFloat.o: RulesInstruction/RulesInstructionFloat.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionFloat.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesInstructionInteger.o: RulesInstruction/RulesInstructionInteger.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionInteger.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesInstructionLogic.o: RulesInstruction/RulesInstructionLogic.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionLogic.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesInstructionMiscellaneous.o: \
 RulesInstruction/RulesInstructionMiscellaneous.cpp \
 RulesInstruction/../DataTypes.h RulesInstruction/../SubString.h \
 RulesInstruction/../helper/helperMixed.h RulesInstruction/../stdafx.h \
 RulesInstruction/../Cubin.h RulesInstruction/../DataTypes.h \
 RulesInstruction/../GlobalVariables.h RulesInstruction/../helper.h \
 RulesInstruction/../helper/helperMixed.h \
 RulesInstruction/../helper/helperParse.h \
 RulesInstruction/../helper/helperCubin.h \
 RulesInstruction/../helper/../Cubin.h \
 RulesInstruction/../helper/helperException.h \
 RulesInstruction/../SpecificParsers.h RulesInstruction/stdafx.h \
 RulesInstruction/RulesInstructionMiscellaneous.h \
 RulesInstruction/../RulesModifier.h \
 RulesInstruction/../RulesModifier/RulesModifierDataMovement.h \
 RulesInstruction/../RulesModifier/RulesModifierInteger.h \
 RulesInstruction/../RulesModifier/RulesModifierFloat.h \
 RulesInstruction/../RulesModifier/RulesModifierConversion.h \
 RulesInstruction/../RulesModifier/RulesModifierCommon.h \
 RulesInstruction/../RulesModifier/RulesModifierExecution.h \
 RulesInstruction/../RulesModifier/RulesModifierLogic.h \
 RulesInstruction/../RulesModifier/RulesModifierOthers.h \
 RulesInstruction/../RulesOperand.h \
 RulesInstruction/../RulesOperand/RulesOperandConstant.h \
 RulesInstruction/../RulesOperand/RulesOperandRegister.h \
 RulesInstruction/../RulesOperand/RulesOperandMemory.h \
 RulesInstruction/../RulesOperand/RulesOperandComposite.h \
 RulesInstruction/../RulesOperand/RulesOperandOthers.h
RulesOperandComposite.o: RulesOperand/RulesOperandComposite.cpp \
 RulesOperand/../DataTypes.h RulesOperand/../SubString.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../DataTypes.h \
 RulesOperand/../Cubin.h RulesOperand/../helper/helperMixed.h \
 RulesOperand/../stdafx.h RulesOperand/../GlobalVariables.h \
 RulesOperand/../helper.h RulesOperand/../helper/helperMixed.h \
 RulesOperand/../helper/helperParse.h \
 RulesOperand/../helper/helperCubin.h RulesOperand/../helper/../Cubin.h \
 RulesOperand/../helper/helperException.h \
 RulesOperand/../SpecificParsers.h RulesOperand/stdafx.h \
 RulesOperand/../RulesOperand.h \
 RulesOperand/../RulesOperand/RulesOperandConstant.h \
 RulesOperand/../RulesOperand/RulesOperandRegister.h \
 RulesOperand/../RulesOperand/RulesOperandMemory.h \
 RulesOperand/../RulesOperand/RulesOperandComposite.h \
 RulesOperand/../RulesOperand/RulesOperandOthers.h \
 RulesOperand/RulesOperandRegister.h RulesOperand/RulesOperandConstant.h \
 RulesOperand/RulesOperandComposite.h
RulesOperandConstant.o: RulesOperand/RulesOperandConstant.cpp \
 RulesOperand/../DataTypes.h RulesOperand/../SubString.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../DataTypes.h \
 RulesOperand/../Cubin.h RulesOperand/../stdafx.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../helper.h \
 RulesOperand/../helper/helperMixed.h \
 RulesOperand/../helper/helperParse.h \
 RulesOperand/../helper/helperCubin.h RulesOperand/../helper/../Cubin.h \
 RulesOperand/../helper/helperException.h \
 RulesOperand/../SpecificParsers.h RulesOperand/stdafx.h \
 RulesOperand/../RulesOperand.h \
 RulesOperand/../RulesOperand/RulesOperandConstant.h \
 RulesOperand/../RulesOperand/RulesOperandRegister.h \
 RulesOperand/../RulesOperand/RulesOperandMemory.h \
 RulesOperand/../RulesOperand/RulesOperandComposite.h \
 RulesOperand/../RulesOperand/RulesOperandOthers.h \
 RulesOperand/RulesOperandConstant.h
RulesOperandMemory.o: RulesOperand/RulesOperandMemory.cpp \
 RulesOperand/../DataTypes.h RulesOperand/../SubString.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../DataTypes.h \
 RulesOperand/../Cubin.h RulesOperand/../stdafx.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../helper.h \
 RulesOperand/../helper/helperMixed.h \
 RulesOperand/../helper/helperParse.h \
 RulesOperand/../helper/helperCubin.h RulesOperand/../helper/../Cubin.h \
 RulesOperand/../helper/helperException.h \
 RulesOperand/../SpecificParsers.h RulesOperand/stdafx.h \
 RulesOperand/../RulesOperand.h \
 RulesOperand/../RulesOperand/RulesOperandConstant.h \
 RulesOperand/../RulesOperand/RulesOperandRegister.h \
 RulesOperand/../RulesOperand/RulesOperandMemory.h \
 RulesOperand/../RulesOperand/RulesOperandComposite.h \
 RulesOperand/../RulesOperand/RulesOperandOthers.h \
 RulesOperand/RulesOperandMemory.h
RulesOperandOthers.o: RulesOperand/RulesOperandOthers.cpp \
 RulesOperand/../DataTypes.h RulesOperand/../SubString.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../DataTypes.h \
 RulesOperand/../Cubin.h RulesOperand/../stdafx.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../helper.h \
 RulesOperand/../helper/helperMixed.h \
 RulesOperand/../helper/helperParse.h \
 RulesOperand/../helper/helperCubin.h RulesOperand/../helper/../Cubin.h \
 RulesOperand/../helper/helperException.h \
 RulesOperand/../SpecificParsers.h RulesOperand/stdafx.h \
 RulesOperand/../RulesOperand.h \
 RulesOperand/../RulesOperand/RulesOperandConstant.h \
 RulesOperand/../RulesOperand/RulesOperandRegister.h \
 RulesOperand/../RulesOperand/RulesOperandMemory.h \
 RulesOperand/../RulesOperand/RulesOperandComposite.h \
 RulesOperand/../RulesOperand/RulesOperandOthers.h \
 RulesOperand/RulesOperandComposite.h RulesOperand/RulesOperandRegister.h \
 RulesOperand/RulesOperandOthers.h
RulesOperandRegister.o: RulesOperand/RulesOperandRegister.cpp \
 RulesOperand/../DataTypes.h RulesOperand/../SubString.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../DataTypes.h \
 RulesOperand/../Cubin.h RulesOperand/../stdafx.h \
 RulesOperand/../GlobalVariables.h RulesOperand/../helper.h \
 RulesOperand/../helper/helperMixed.h \
 RulesOperand/../helper/helperParse.h \
 RulesOperand/../helper/helperCubin.h RulesOperand/../helper/../Cubin.h \
 RulesOperand/../helper/helperException.h \
 RulesOperand/../SpecificParsers.h RulesOperand/stdafx.h \
 RulesOperand/RulesOperandRegister.h RulesOperand/../RulesOperand.h \
 RulesOperand/../RulesOperand/RulesOperandConstant.h \
 RulesOperand/../RulesOperand/RulesOperandRegister.h \
 RulesOperand/../RulesOperand/RulesOperandMemory.h \
 RulesOperand/../RulesOperand/RulesOperandComposite.h \
 RulesOperand/../RulesOperand/RulesOperandOthers.h
RulesModifierCommon.o: RulesModifier/RulesModifierCommon.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierCommon.h
RulesModifierConversion.o: RulesModifier/RulesModifierConversion.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierConversion.h RulesModifier/../RulesModifier.h \
 RulesModifier/../RulesModifier/RulesModifierDataMovement.h \
 RulesModifier/../RulesModifier/RulesModifierInteger.h \
 RulesModifier/../RulesModifier/RulesModifierFloat.h \
 RulesModifier/../RulesModifier/RulesModifierConversion.h \
 RulesModifier/../RulesModifier/RulesModifierCommon.h \
 RulesModifier/../RulesModifier/RulesModifierExecution.h \
 RulesModifier/../RulesModifier/RulesModifierLogic.h \
 RulesModifier/../RulesModifier/RulesModifierOthers.h \
 RulesModifier/../RulesOperand.h \
 RulesModifier/../RulesOperand/RulesOperandConstant.h \
 RulesModifier/../RulesOperand/RulesOperandRegister.h \
 RulesModifier/../RulesOperand/RulesOperandMemory.h \
 RulesModifier/../RulesOperand/RulesOperandComposite.h \
 RulesModifier/../RulesOperand/RulesOperandOthers.h
RulesModifierDataMovement.o: RulesModifier/RulesModifierDataMovement.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierDataMovement.h
RulesModifierExecution.o: RulesModifier/RulesModifierExecution.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierExecution.h RulesModifier/../RulesOperand.h \
 RulesModifier/../RulesOperand/RulesOperandConstant.h \
 RulesModifier/../RulesOperand/RulesOperandRegister.h \
 RulesModifier/../RulesOperand/RulesOperandMemory.h \
 RulesModifier/../RulesOperand/RulesOperandComposite.h \
 RulesModifier/../RulesOperand/RulesOperandOthers.h \
 RulesModifier/../RulesModifier.h \
 RulesModifier/../RulesModifier/RulesModifierDataMovement.h \
 RulesModifier/../RulesModifier/RulesModifierInteger.h \
 RulesModifier/../RulesModifier/RulesModifierFloat.h \
 RulesModifier/../RulesModifier/RulesModifierConversion.h \
 RulesModifier/../RulesModifier/RulesModifierCommon.h \
 RulesModifier/../RulesModifier/RulesModifierExecution.h \
 RulesModifier/../RulesModifier/RulesModifierLogic.h \
 RulesModifier/../RulesModifier/RulesModifierOthers.h
RulesModifierFloat.o: RulesModifier/RulesModifierFloat.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierFloat.h
RulesModifierInteger.o: RulesModifier/RulesModifierInteger.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierInteger.h
RulesModifierLogic.o: RulesModifier/RulesModifierLogic.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierLogic.h
RulesModifierOthers.o: RulesModifier/RulesModifierOthers.cpp \
 RulesModifier/../DataTypes.h RulesModifier/../SubString.h \
 RulesModifier/../helper/helperMixed.h RulesModifier/../stdafx.h \
 RulesModifier/../Cubin.h RulesModifier/../DataTypes.h \
 RulesModifier/../GlobalVariables.h RulesModifier/../helper.h \
 RulesModifier/../helper/helperMixed.h \
 RulesModifier/../helper/helperParse.h \
 RulesModifier/../helper/helperCubin.h RulesModifier/../helper/../Cubin.h \
 RulesModifier/../helper/helperException.h \
 RulesModifier/../SpecificParsers.h RulesModifier/stdafx.h \
 RulesModifier/RulesModifierOthers.h
