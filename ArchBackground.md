## Basic Information ##
According to CUDA programming guide 4.0 Section F.4.1, for each Multiprocessor (MP) of a Compute Capability 2.0 device:
  * There are 32 cuda cores, divided into 2 groups of 16, for integer/floating point operations,
  * 4 special function units for single-precision floating-point transcendental functions,
  * 2 warp schedulers
  * One instruction is issued per scheduler per clock
  * First scheduler is in charge of warps with odd IDs. Second scheduler is in charge of warps with even IDs.
  * A warp scheduler issues instruction to 16 CUDA cores at a time. For an instruction to be executed for an entire warp, the scheduler issues the instruction over 2 CUDA core clocks(1 scheduler clock, as scheduler works at half the frequency of the CUDA cores).