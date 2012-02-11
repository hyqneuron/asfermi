#ifndef LIBASFERMI_H
#define LIBASFERMI_H

#include <stdint.h>

// Fermi Assembler by hyqneuron: library mode interface

#ifdef __cplusplus
extern "C"
{
#endif

// Emit cubin ELF containing Fermi instructions for the
// given source code, compute capability and ELF bitness
// (0 - 32-bit, 1 - 64-bit).
char* asfermi_encode_cubin(char* source, int cc, int elf64bit, size_t* szcubin);

// Emit plain array containing Fermi instructions for the
// given source code and compute capability.
char* asfermi_encode_opcodes(char* source, int cc, size_t* szopcodes);

#ifdef __cplusplus
}
#endif

#endif // LIBASFERMI_H

