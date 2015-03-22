# Project Plans #

## Project Stages ##
This project involves two stages. First is the making of the assembler. Second is the probing of various detailed features of the GF1xx architectures.

### Assembler ###
The major parts of the assembler are already done.

A few more things to do:
  1. Write tests and debug

Other possible future features:
  1. Support a subset of C to ease coding
  1. Macros (+ a bit of scripting to help with unrolling)
  1. Register renaming and instruction reordering

### Microbenchmarking ###
#### Instruction-related ####
  * Instruction latencies
  * Warp scheduling
    * scheduling pattern
    * replay
    * Divergence behaviour
    * Processing Unit Group characteristics
  * Register file
    * row size
    * ports
    * allocation
  * Instruction cache
    * size, associativity, prefetch pattern

#### Memory-related ####
  * Latency
    * Various types of caches and memories
    * Memoryfence
    * Atomic
  * Cache associativity
  * Cache eviction policy
  * L2 structure
  * TLB location
  * Memory controller
    * Hashing
    * Dual-channel
    * Service order
  * Cache consistency across kernel launches

#### Others ####
  * Kernel launch overhead
  * Block scheduling overhead
  * Block scheduling pattern

---

### Sidelines ###
#### Utilities ####
See [Utilities](Utilities.md).


<a href='Hidden comment: 
build a utility that permute selected bits. Each instance of permutation gets injected into an individual file. Generate as many files as there are instances of the permutation and test run all the files to confirm the validity of the instructions. Take down the code of the permutation for the cases that are validly executed. Generate a .txt table with all the permuted code and a string to represent their validity. Then sort them according to specific bits to view their validity. Sort them using excel or perhaps build another utility to sort and identify the modifiers.
1. say 10 bits to be permuted
2. 1024 cubins, 1024 runs
3. A file showing the permuted binary digits of all files and their corresponding validity
4. Sort the file in 3 to find out valid independent and compound modifier bits.
5. Probe the modifier names using cuobjdump.
'></a>