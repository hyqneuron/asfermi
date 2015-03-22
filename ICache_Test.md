

Note: the information below is not confirmative. I'm using this site to take note

### Interpretation ###
  * instruction prefetch size is 64 bytes (8 instructions)

### Test1 ###
#### code ####
```
!Machine 64
!Kernel k
!Param 8 2
!Param 4

S2R R0, SR_ClockLo
S2R R1, SR_ClockLo
S2R R2, SR_ClockLo
S2R R3, SR_ClockLo
S2R R4, SR_ClockLo
S2R R5, SR_ClockLo
S2R R6, SR_ClockLo
S2R R7, SR_ClockLo
S2R R8, SR_ClockLo
S2R R9, SR_ClockLo
S2R R10, SR_ClockLo
S2R R11, SR_ClockLo
S2R R12, SR_ClockLo
S2R R13, SR_ClockLo
S2R R14, SR_ClockLo
S2R R15, SR_ClockLo
S2R R16, SR_ClockLo
S2R R17, SR_ClockLo
S2R R18, SR_ClockLo
S2R R19, SR_ClockLo
S2R R20, SR_ClockLo
S2R R21, SR_ClockLo
S2R R22, SR_ClockLo
S2R R23, SR_ClockLo
S2R R24, SR_ClockLo
S2R R25, SR_ClockLo
S2R R26, SR_ClockLo
S2R R27, SR_ClockLo
S2R R28, SR_ClockLo
S2R R29, SR_ClockLo
S2R R30, SR_ClockLo
S2R R31, SR_ClockLo
S2R R32, SR_ClockLo
S2R R33, SR_ClockLo
S2R R34, SR_ClockLo
MOV R50, c[0x0][0x20]
MOV R51, c[0x0][0x24]
ST.E [R50], R0
ST.E [R50+0x4], R1
ST.E [R50+0x8], R2
ST.E [R50+0xc], R3
ST.E [R50+0x10], R4
ST.E [R50+0x14], R5
ST.E [R50+0x18], R6
ST.E [R50+0x1c], R7
ST.E [R50+0x20], R8
ST.E [R50+0x24], R9
ST.E [R50+0x28], R10
ST.E [R50+0x2c], R11
ST.E [R50+0x30], R12
ST.E [R50+0x34], R13
ST.E [R50+0x38], R14
ST.E [R50+0x3c], R15
ST.E [R50+0x40], R16
ST.E [R50+0x44], R17
ST.E [R50+0x48], R18
ST.E [R50+0x4c], R19
ST.E [R50+0x50], R20
ST.E [R50+0x54], R21
ST.E [R50+0x58], R22
ST.E [R50+0x5c], R23
ST.E [R50+0x60], R24
ST.E [R50+0x64], R25
ST.E [R50+0x68], R26
ST.E [R50+0x6c], R27
ST.E [R50+0x70], R28
ST.E [R50+0x74], R29
ST.E [R50+0x78], R30
ST.E [R50+0x7c], R31
ST.E [R50+0x80], R32
ST.E [R50+0x84], R33
ST.E [R50+0x88], R34
EXIT
!EndKernel


```

#### result ####
```
length=35
tcount=1
i=0, j=0, output=1469777
i=1, j=0, output=1469780 //normal increment of 3
i=2, j=0, output=1469783
i=3, j=0, output=1469786
i=4, j=0, output=1469789
i=5, j=0, output=1469792
i=6, j=0, output=1469795
i=7, j=0, output=1469798
i=8, j=0, output=1469805 //abnormal increment of 7
i=9, j=0, output=1469808
i=10, j=0, output=1469811
i=11, j=0, output=1469814
i=12, j=0, output=1469817
i=13, j=0, output=1469820
i=14, j=0, output=1469823
i=15, j=0, output=1469826
i=16, j=0, output=1469833//abnormal
i=17, j=0, output=1469836
i=18, j=0, output=1469839
i=19, j=0, output=1469842
i=20, j=0, output=1469845
i=21, j=0, output=1469848
i=22, j=0, output=1469851
i=23, j=0, output=1469854
i=24, j=0, output=1469861//abnormal
i=25, j=0, output=1469864
i=26, j=0, output=1469867
i=27, j=0, output=1469870
i=28, j=0, output=1469873
i=29, j=0, output=1469876
i=30, j=0, output=1469879
i=31, j=0, output=1469882
i=32, j=0, output=1469929//huge jump at 33rd instruction
i=33, j=0, output=1469932
i=34, j=0, output=1469935


```