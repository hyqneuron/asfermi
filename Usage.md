Usage:
```
asfermi sourcefile [Options [option arguments]]
```
Source file must be the first command-line option. However, it could be replaced by a -I option specified below.

For how a source file should look like, please see [CodeExample](CodeExample.md).

Options:
  * <pre>-I "instruction"</pre>This can be used to replace the inputfile. A single line of instruction, surrounded by double quotation marks, will be processed as source input. Note that comments and directives are not supported in this mode.
  * <pre>-o outputfile</pre>Output cubin to the specified file.
  * <pre>-r target_cubin .text.kernel_name kernel_offset</pre>Replace the opcodes in specified location of a kernel in a specified cubin file with the assembled opcodes. Note that the kernel name must be prefixed with .text. to indicate the complete cubin section name.
  * <pre>-sm_20</pre>output cubin for architecture sm\_20. This is the default architecture assumed by asfermi.
  * <pre>-sm_21</pre>output cubin for architecture sm\_21.
  * <pre>-32</pre>output 32-bit cubin. This is the default behaviour.
  * <pre>-64</pre>output 64-bit cubin.
  * <pre>-SelfDebug</pre>throw unhandled exception when things go wrong. For debugging of asfermi only.