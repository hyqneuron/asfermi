<a href='Hidden comment: 
#summary Organisation of code

This page is written by hyq.neuron for the other asfermi team members.

== General Organisation ==
The entire body of the code can be divided into 4 sections:
# Body (asfermi.cpp)
# DataTypes (SubString.h, DataTypes.h, Cubin.h)
# Rules (Rules*.h)
# Helper functions (helper*.h)

=== Body ===
asfermi.cpp contains these functions:
# main: program entry point
# ProcessCommandsAndReadSource: processes command line options and read source file
# Initialize: initializes all the rules
# OrganiseRules: function called by Initialize to help with the initialization process.
# WriteToCubinDirectOutput: function called by main for cubin output
# WriteToCubinReplace: function called by main for cubin output(replace mode)

Things certainly



== Things to take note of ==
# source code: entirety of the source code is stored in a global char array, csSource(GlobalVariables.h). The length of csSource is the length of the source file + 1 because sometimes asfermi may need to append a null character at the end to indicate an end-of-line.
# Lines: the source file is broken into lines, which are stored in the structure Line. However, a Line structure itself does not contain the string. Instead, it contains a SubString structure, LineString.
# SubString: SubString contains a char pointer, Start, to the beginning of the sub-string. It also contains an integer number, Length, that records the length of this sub-string. Individual characters can be accessed using the square bracket notation.



Basic flow

Suppose we have the following source file
```
!Kernel kernel1
MOV R1, c [0x1] [0x100];
EXIT;
!EndKernel
```
And we assemble it with the following command line
```
asfermi example.txt -o example.cubin
```
Here"s the sequence of things that asfermi will do:
# Process command line. (asfermi.cpp, ProcessCommandsAndReadSource)
# Read source file into csSource, a global char array. (asfermi.cpp, ProcessCommandsAndReadSource)
# csSource is then divided into Lines, and comments are removed from lines at this stage (helperMain.h, hpReadSource)
# Initialize instruction and directive rules (asfermi.cpp, Initialize)
# Add all instruction rules and directive rules to the lists csInstructionRulePrepList and csDirectiveRulePrepList respectively. (Initialize)
# Compute index for each instruction based on its instruction name (asfermi.cpp, OrganiseRules)
# Sort the indices and copy instruction rules in ascending order of their indices into the array csInstructionRules. (asfermi.cpp, OrganiseRules)
# Same goes for the directive rules
# Call the csMasterParser"s Parse function, which iterates through all the lines
# call the csLineParser->Parse(line) for each line.

'></a>