

### Interpretation ###
Current results with S2R suggest:
  * S2R has a throughput of 64 instructions per scheduler clock(SK) per Multiprocessor (MP)
  * The scheduler fetches 16 bytes (2 8-byte instructions. This is probably the row size of the instruction cache) at a time.
  * Even-numbered warps are often 1 step slower than odd-numbered warps. This reduces Processing Unit Group contention.
  * Minimum of 6 warps per SM is needed to achieve peak throughput (suggested by Sylvain Collange)

Result of Test1 (with 16 warps, showing only the even-numbered warps, numbers in the following tables are the normalised clock numbers):
![https://lh5.googleusercontent.com/-LNlcS40ML2g/TkU7X9Hcc7I/AAAAAAAAAPo/FHf7s6WS7SQ/s800/S2R%252520result2.jpg](https://lh5.googleusercontent.com/-LNlcS40ML2g/TkU7X9Hcc7I/AAAAAAAAAPo/FHf7s6WS7SQ/s800/S2R%252520result2.jpg)

Result of Test2 (with 16 warps, showing the even-numbered warps only):

![https://lh3.googleusercontent.com/-FW0EI9Nq5Qk/TkU4irz7cfI/AAAAAAAAAPU/nsKBdtu8rgM/s800/S2R%252520result.jpg](https://lh3.googleusercontent.com/-FW0EI9Nq5Qk/TkU4irz7cfI/AAAAAAAAAPU/nsKBdtu8rgM/s800/S2R%252520result.jpg)

Result of test 1, with 2,4 and 6 warps respectively. Only even-numbered warps are being shown.

https://lh5.googleusercontent.com/-wLpcDZ6K9Oc/TkYLtM5uVhI/AAAAAAAAAQw/fzbQieI8Z-g/s800/S2R%2525202%252520warps.JPG
https://lh6.googleusercontent.com/-yGhQVJ7qUZU/TkYLvn1RaDI/AAAAAAAAAQw/7gHOHOJF9cY/s800/S2R%2525204%252520warps.JPG
https://lh6.googleusercontent.com/-EJVv7YzlFH8/TkYLwtvd2YI/AAAAAAAAAQw/Rac0T3gAyOU/s800/S2R%2525206%252520warps.JPG

### Test1 ###
#### code ####
```
!Machine 64
!Kernel k
!Param 8 2
!Param 4

//Start off by loading SR_ClockLo into registers
S2R     R0,     SR_ClockLo;
S2R     R1,     SR_ClockLo;
S2R     R2,     SR_ClockLo;
S2R     R3,     SR_ClockLo;
S2R     R10,    SR_ClockLo;
S2R     R11,    SR_ClockLo;

//Store the clock numbers
//Load first parameter into R4&R5 (64-bit addressing)
MOV R4, c[0X0][0X20]
MOV R5, c[0X0][0X24]
S2R R6, SR_Tid_X;
SHR R7, R6, 5; //WARPID

//R4 = base addr + WarpId*0x18
IMAD  R4, R7, 0x18,  R4; 
LOP.AND R6,R6, 0X1F;//LANEID
ISETP.EQ P0, P7, R6, RZ; //P0 = laneId == zero
@!P0 EXIT; //not zero, quit
ST.E [R4+0X0], R0;
ST.E [R4+0X4], R1
ST.E [R4+0X8], R2;
ST.E [R4+0XC], R3;
ST.E [R4+0X10], R10;
ST.E [R4+0x14], R11;
EXIT;
!EndKernel
```

#### result ####

```
length = 192 //number of unsigned integers written
tcount=1024  //number of threads

Device clock rate:1544000
time:0.025472

i=0, output=2981802815
i=1, output=2981802818
i=2, output=2981802847
i=3, output=2981802850
i=4, output=2981802879
i=5, output=2981802882
i=6, output=2981802814
i=7, output=2981802817
i=8, output=2981802846
i=9, output=2981802849
i=10, output=2981802878
i=11, output=2981802881
i=12, output=2981802817
i=13, output=2981802820
i=14, output=2981802849
i=15, output=2981802852
i=16, output=2981802881
i=17, output=2981802884
i=18, output=2981802816
i=19, output=2981802819
i=20, output=2981802848
i=21, output=2981802851
i=22, output=2981802880
i=23, output=2981802883
i=24, output=2981802819
i=25, output=2981802822
i=26, output=2981802851
i=27, output=2981802854
i=28, output=2981802883
i=29, output=2981802886
i=30, output=2981802818
i=31, output=2981802821
i=32, output=2981802850
i=33, output=2981802853
i=34, output=2981802882
i=35, output=2981802885
i=36, output=2981802821
i=37, output=2981802824
i=38, output=2981802853
i=39, output=2981802856
i=40, output=2981802885
i=41, output=2981802888
i=42, output=2981802820
i=43, output=2981802823
i=44, output=2981802852
i=45, output=2981802855
i=46, output=2981802884
i=47, output=2981802887
i=48, output=2981802823
i=49, output=2981802826
i=50, output=2981802855
i=51, output=2981802858
i=52, output=2981802887
i=53, output=2981802890
i=54, output=2981802822
i=55, output=2981802825
i=56, output=2981802854
i=57, output=2981802857
i=58, output=2981802886
i=59, output=2981802889
i=60, output=2981802825
i=61, output=2981802828
i=62, output=2981802857
i=63, output=2981802860
i=64, output=2981802889
i=65, output=2981802892
i=66, output=2981802824
i=67, output=2981802827
i=68, output=2981802856
i=69, output=2981802859
i=70, output=2981802888
i=71, output=2981802891
i=72, output=2981802827
i=73, output=2981802830
i=74, output=2981802859
i=75, output=2981802862
i=76, output=2981802891
i=77, output=2981802894
i=78, output=2981802826
i=79, output=2981802829
i=80, output=2981802858
i=81, output=2981802861
i=82, output=2981802890
i=83, output=2981802893
i=84, output=2981802829
i=85, output=2981802832
i=86, output=2981802861
i=87, output=2981802864
i=88, output=2981802893
i=89, output=2981802896
i=90, output=2981802828
i=91, output=2981802831
i=92, output=2981802860
i=93, output=2981802863
i=94, output=2981802892
i=95, output=2981802895
i=96, output=2981802831
i=97, output=2981802834
i=98, output=2981802863
i=99, output=2981802866
i=100, output=2981802895
i=101, output=2981802898
i=102, output=2981802830
i=103, output=2981802833
i=104, output=2981802862
i=105, output=2981802865
i=106, output=2981802894
i=107, output=2981802897
i=108, output=2981802833
i=109, output=2981802836
i=110, output=2981802865
i=111, output=2981802868
i=112, output=2981802897
i=113, output=2981802900
i=114, output=2981802832
i=115, output=2981802835
i=116, output=2981802864
i=117, output=2981802867
i=118, output=2981802896
i=119, output=2981802899
i=120, output=2981802835
i=121, output=2981802838
i=122, output=2981802867
i=123, output=2981802870
i=124, output=2981802899
i=125, output=2981802902
i=126, output=2981802834
i=127, output=2981802837
i=128, output=2981802866
i=129, output=2981802869
i=130, output=2981802898
i=131, output=2981802901
i=132, output=2981802837
i=133, output=2981802840
i=134, output=2981802869
i=135, output=2981802872
i=136, output=2981802901
i=137, output=2981802904
i=138, output=2981802836
i=139, output=2981802839
i=140, output=2981802868
i=141, output=2981802871
i=142, output=2981802900
i=143, output=2981802903
i=144, output=2981802839
i=145, output=2981802842
i=146, output=2981802871
i=147, output=2981802874
i=148, output=2981802903
i=149, output=2981802906
i=150, output=2981802838
i=151, output=2981802841
i=152, output=2981802870
i=153, output=2981802873
i=154, output=2981802902
i=155, output=2981802905
i=156, output=2981802841
i=157, output=2981802844
i=158, output=2981802873
i=159, output=2981802876
i=160, output=2981802905
i=161, output=2981802908
i=162, output=2981802840
i=163, output=2981802843
i=164, output=2981802872
i=165, output=2981802875
i=166, output=2981802904
i=167, output=2981802907
i=168, output=2981802843
i=169, output=2981802846
i=170, output=2981802875
i=171, output=2981802878
i=172, output=2981802907
i=173, output=2981802910
i=174, output=2981802842
i=175, output=2981802845
i=176, output=2981802874
i=177, output=2981802877
i=178, output=2981802906
i=179, output=2981802909
i=180, output=2981802845
i=181, output=2981802848
i=182, output=2981802877
i=183, output=2981802880
i=184, output=2981802909
i=185, output=2981802912
i=186, output=2981802844
i=187, output=2981802847
i=188, output=2981802876
i=189, output=2981802879
i=190, output=2981802908
i=191, output=2981802911
```

### Test2 ###
#### code ####
```
!Machine 64

!Kernel k
!Param 8 2
!Param 4

NOP;
S2R     R0,     SR_ClockLo;
S2R     R1,     SR_ClockLo;
S2R     R2,     SR_ClockLo;
S2R     R3,     SR_ClockLo;
S2R     R10,    SR_ClockLo;
S2R     R11,    SR_ClockLo;


//addr = warpID * rowsize + base addr
MOV R4, c[0X0][0X20]
MOV R5, c[0X0][0X24]
S2R R6, SR_Tid_X
SHR R7, R6, 5 //WARPID
IMAD  R4, R7, 0x18,  R4 // *8
LOP.AND R6,R6, 0X1F;//LANEID
ISETP.EQ P0, P7, R6, RZ;
@!P0 EXIT;
ST.E [R4+0X0], R0;
ST.E [R4+0X4], R1
ST.E [R4+0X8], R2;
ST.E [R4+0XC], R3;
ST.E [R4+0X10], R10;
ST.E [R4+0x14], R11;
EXIT;
!EndKernel
```
#### result ####
```
length = 192
tcount=1024
Device clock rate:1544000
time:0.024864

i=0, output=2935324134
i=1, output=2935324163
i=2, output=2935324166
i=3, output=2935324195
i=4, output=2935324198
i=5, output=2935324227
i=6, output=2935324133
i=7, output=2935324162
i=8, output=2935324165
i=9, output=2935324194
i=10, output=2935324197
i=11, output=2935324226
i=12, output=2935324136
i=13, output=2935324165
i=14, output=2935324168
i=15, output=2935324197
i=16, output=2935324200
i=17, output=2935324229
i=18, output=2935324135
i=19, output=2935324164
i=20, output=2935324167
i=21, output=2935324196
i=22, output=2935324199
i=23, output=2935324228
i=24, output=2935324138
i=25, output=2935324167
i=26, output=2935324170
i=27, output=2935324199
i=28, output=2935324202
i=29, output=2935324231
i=30, output=2935324137
i=31, output=2935324166
i=32, output=2935324169
i=33, output=2935324198
i=34, output=2935324201
i=35, output=2935324230
i=36, output=2935324140
i=37, output=2935324169
i=38, output=2935324172
i=39, output=2935324201
i=40, output=2935324204
i=41, output=2935324233
i=42, output=2935324139
i=43, output=2935324168
i=44, output=2935324171
i=45, output=2935324200
i=46, output=2935324203
i=47, output=2935324232
i=48, output=2935324142
i=49, output=2935324171
i=50, output=2935324174
i=51, output=2935324203
i=52, output=2935324206
i=53, output=2935324235
i=54, output=2935324141
i=55, output=2935324170
i=56, output=2935324173
i=57, output=2935324202
i=58, output=2935324205
i=59, output=2935324234
i=60, output=2935324144
i=61, output=2935324173
i=62, output=2935324176
i=63, output=2935324205
i=64, output=2935324208
i=65, output=2935324237
i=66, output=2935324143
i=67, output=2935324172
i=68, output=2935324175
i=69, output=2935324204
i=70, output=2935324207
i=71, output=2935324236
i=72, output=2935324146
i=73, output=2935324175
i=74, output=2935324178
i=75, output=2935324207
i=76, output=2935324210
i=77, output=2935324239
i=78, output=2935324145
i=79, output=2935324174
i=80, output=2935324177
i=81, output=2935324206
i=82, output=2935324209
i=83, output=2935324238
i=84, output=2935324148
i=85, output=2935324177
i=86, output=2935324180
i=87, output=2935324209
i=88, output=2935324212
i=89, output=2935324241
i=90, output=2935324147
i=91, output=2935324176
i=92, output=2935324179
i=93, output=2935324208
i=94, output=2935324211
i=95, output=2935324240
i=96, output=2935324150
i=97, output=2935324179
i=98, output=2935324182
i=99, output=2935324211
i=100, output=2935324214
i=101, output=2935324243
i=102, output=2935324149
i=103, output=2935324178
i=104, output=2935324181
i=105, output=2935324210
i=106, output=2935324213
i=107, output=2935324242
i=108, output=2935324152
i=109, output=2935324181
i=110, output=2935324184
i=111, output=2935324213
i=112, output=2935324216
i=113, output=2935324245
i=114, output=2935324151
i=115, output=2935324180
i=116, output=2935324183
i=117, output=2935324212
i=118, output=2935324215
i=119, output=2935324244
i=120, output=2935324154
i=121, output=2935324183
i=122, output=2935324186
i=123, output=2935324215
i=124, output=2935324218
i=125, output=2935324247
i=126, output=2935324153
i=127, output=2935324182
i=128, output=2935324185
i=129, output=2935324214
i=130, output=2935324217
i=131, output=2935324246
i=132, output=2935324156
i=133, output=2935324185
i=134, output=2935324188
i=135, output=2935324217
i=136, output=2935324220
i=137, output=2935324249
i=138, output=2935324155
i=139, output=2935324184
i=140, output=2935324187
i=141, output=2935324216
i=142, output=2935324219
i=143, output=2935324248
i=144, output=2935324158
i=145, output=2935324187
i=146, output=2935324190
i=147, output=2935324219
i=148, output=2935324222
i=149, output=2935324251
i=150, output=2935324157
i=151, output=2935324186
i=152, output=2935324189
i=153, output=2935324218
i=154, output=2935324221
i=155, output=2935324250
i=156, output=2935324160
i=157, output=2935324189
i=158, output=2935324192
i=159, output=2935324221
i=160, output=2935324224
i=161, output=2935324253
i=162, output=2935324159
i=163, output=2935324188
i=164, output=2935324191
i=165, output=2935324220
i=166, output=2935324223
i=167, output=2935324252
i=168, output=2935324162
i=169, output=2935324191
i=170, output=2935324194
i=171, output=2935324223
i=172, output=2935324226
i=173, output=2935324255
i=174, output=2935324161
i=175, output=2935324190
i=176, output=2935324193
i=177, output=2935324222
i=178, output=2935324225
i=179, output=2935324254
i=180, output=2935324164
i=181, output=2935324193
i=182, output=2935324196
i=183, output=2935324225
i=184, output=2935324228
i=185, output=2935324257
i=186, output=2935324163
i=187, output=2935324192
i=188, output=2935324195
i=189, output=2935324224
i=190, output=2935324227
i=191, output=2935324256
```

### Test3 ###
#### code ####
```
```