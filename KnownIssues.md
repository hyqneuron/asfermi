## Known Issues ##
  1. MUFU.Rxx64H currently does not cause the register count to increase by 2. Whether increasing register count by 2 instead of by 1 is necessary or not is to be confirmed.
  1. Error messages are reused too often and some errors will not trigger the correct error message.
  1. No warning is given when the parsing of some floating point/integer expressions using atoi(), atol() or atof() produce errors.

## Others ##
  1. Memory leaks are not being dealt with.
  1. ~~In a recent test asfermi assembled 10,000 instructions in less than a second. But as the number of instruction increases, the time involved in assembly increases exponentially. If you intend to use asfermi to assemble huge kernels perhaps you'll want to email me first. Currently resolving the performance issue with large source files is not on the to-do list.~~ Fixed. Now processing time is linear. With a regular HDD most of the time is spent on file I/O.