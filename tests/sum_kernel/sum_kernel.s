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
MOV R5, c [0x0] [0x24]; // TODO: IADD.X R5, RZ
LD.E R1, [R4];
FADD R0, R0, R1;
IADD R4, R3, c [0x0] [0x30]; // TODO: R4.CC
MOV R5, c [0x0] [0x24]; // TODO: IADD.X R5, RZ
ST.E [R4], R0;
EXIT;
!EndKernel

