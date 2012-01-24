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
!Param 8 1
MOV R0, c[0x0][0x20];
MOV R1, c[0x0][0x24];
MOV R2, c[0x2][0x0];
ST.E [R0], R2;
EXIT;
!EndKernel
