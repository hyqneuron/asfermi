/*
 * Copyright (c) 2011, 2012 by Hou Yunqing and Dmitry Mikushin
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

