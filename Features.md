# Features #
This page contains a list of features that are supported/to be supported. For the meaning of the state numbers (at the beginning of each feature), see [Features#State\_Numbers](Features#State_Numbers.md). For features that can be enabled through the use of directives, please refer to [Directives](Directives.md).

  * 0: Replace opcodes at specified location in a specified kernel of a cubin.
  * 0: Output assembled opcodes directly as a cubin
  * 2: Insert instructions into a specified kernel of a cubin
  * 3: Output cubin with symbolic debugging information
  * 5: Optimization of instructions (for specific architectures)


---

## Extensibility ##
  * 0: Passive Parser, see [Design#Line\_Parser](Design#Line_Parser.md)
  * 0: Active Parser


---

## State Numbers ##
  * **0**: Already supported
  * **1**: Priority level 1, top priority.
  * **2**: Priority level 2
  * **3**: Priority level 3
  * **5**: Not decided
  * **8**: Unlikely to be supported
  * **9**: Not going to be supported