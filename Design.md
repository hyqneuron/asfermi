Note: Information on this page is no longer accurate.





---

## Organisation ##
### General structure ###
  1. Preprocess: process command-line arguments, read input file into Line and check the existing output cubin.
  1. Process: calls the default Master Parser to process the lines.
  1. Postprocess: write assembled opcode into cubin
#### Master Parser ####
The master parser calls the line parser once for each line.

Master parsers are active parsers, which are given a starting line number and are then free to jump between lines and do the necessary processing.
#### Line Parser ####
Note: Currently the default Line Parser is implemented inside the default Master Parser itself. This may change in the future.
```
    if(line is directive)
        directive_parser(line);
    else
    {
        break the line into multiple instructions separated by ';'
        instruction_parser(each instruction);
    }
```

The **Line parser** receives a line, makes a judgement about whether this line is a directive or an instruction, and then
  * Asks the **directive parser** to parse the line, if the line is a directive, or
  * Asks the **instruction parser** to parse the processed parts of the line, if the line is an instruction.

Line parsers are passive parsers, which parse only one line that is given by the caller.

#### Instruction Parser ####
An instruction parser is a passive parser. It receives a single instruction from the caller (line parser), and then analyses the instruction to produce the correct opcode. It then appends the opcode to a list and increase an offset count by the byte length of the opcode.

The default instruction parser, upon receiving the instruction line, would check against a list of known instruction rules and treat the instruction operands in accordance with the rules specified under the corresponding instruction name.

For more details of the instructions, see [SourceFormat](SourceFormat.md)

#### Directive Parser ####
A directive parser is a passive parser. It receives a single command (a single line) from the caller (line parser), and does the corresponding changes to the assembler's states.

It is also in charge of registering the labels found into a list along with its current instruction offset.

The default directive parser, upon receiving the directive line, would check against a list of known directive rules and treat the arguments of the directive in accordance with the rules specified under the corresponding directive name.

For more details of the directives, see [SourceFormat](SourceFormat.md)


---

## Extensibility ##
Default master, line and instruction parsers can be replaced by custom parsers under the request of specific directives. See [Directives#Set\_current\_parsers](Directives#Set_current_parsers.md)

With this feature, asfermi could process non-standard lines of code. This feature could be used to support some high-level style coding.