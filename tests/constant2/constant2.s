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

!Constant2 0x40
!Constant int 0x0 //first argument indicates the type of the constant
                  //second argument indicates the offset of this object 
                  //from the beginning of the .nv.constant2 section
1, 2, 3, 4
!EndConstant

!Constant mixed 0x10 named
0x1001, F-1.3, FH1.909090909090, H10101010101010
!EndConstant

!Kernel kernel
!Param 256 1
MOV R0, c[0x0][0x20];
MOV R1, c[0x0][0x24];
MOV R2, c[0x2][0x0];
ST.E [R0], R2;
EXIT;
!EndKernel
