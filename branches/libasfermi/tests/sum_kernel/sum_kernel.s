/*
 * Copyright (c) 2012 by Dmitry Mikushin
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

// The Fermi assember for the sum_kernel:
// __global__ void sum_kernel ( float * a, float * b, float * c )
// {
//     int idx = threadIdx.x + blockIdx.x * blockDim.x;
//     c [idx] = a [idx] + b [idx];
// }

!Kernel kernel
!Param 8 3
S2R R0, SR_CTAid_X;
S2R R1, SR_Tid_X;
IMAD R3, R0, c [0x0] [0x8], R1;
IMUL R3, R3, 0x4;
IADD R4, R3, c [0x0] [0x20]; // TODO: R4.CC
MOV R5, c [0x0] [0x24]; // TODO: IADD.X R5, RZ
LD.E R0, [R4];
IADD R4, R3, c [0x0] [0x28]; // TODO: R4.CC
MOV R5, c [0x0] [0x2c]; // TODO: IADD.X R5, RZ
LD.E R1, [R4];
FADD R0, R0, R1;
IADD R4, R3, c [0x0] [0x30]; // TODO: R4.CC
MOV R5, c [0x0] [0x34]; // TODO: IADD.X R5, RZ
ST.E [R4], R0;
EXIT;
!EndKernel

