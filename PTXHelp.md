This page is written by hyq.neuron to help other asfermi members get started probing instruction opcodes.


```
.version 2.3
.target sm_20
.address_size 32

.entry InfoKernel (.param.u32 uoutput, .param.f32 foutput, .param.s32 soutput)
{

//I keep a lot of registers of various types so that I don't have to declare registers everytime I start on a new instruction.
	.local.b32 localarray[10];
	.const.b32 constarray[10];

	.reg.u8  russ<15>;
	.reg.u16 rus<15>;
	.reg.u32 ru<15>;
	.reg.u64 rul<15>;

	.reg.s16 rss<15>;
	.reg.s32 rs<15>;
	.reg.s64 rsl<15>;

	.reg.f16 rfs<15>;
	.reg.f32 rf<15>;
	.reg.f64 rfl<15>;

	.reg.b16 rbs<15>;
	.reg.b32 rb<15>;
	.reg.b64 rbl<15>;
	.reg.pred p<5>;

//load  input/output address
	ld.param.u32	ru0, [uoutput]; 

//just load some values into the registers. Without this step ptxas will sometimes skip certain instructions of interest to us
	ld.u32 ru1, [ru0 + 0x0];
	ld.u32 ru2, [ru0 + 0x4];
	ld.u32 ru3, [ru0 + 0x8];
	ld.u32 ru4, [ru0 + 0xc];
	ld.u64 rul5, [ru0 + 0x10];
	ld.s64 rsl6, [ru0 + 0x18];

//Here's the instructions we're probing. The following is written for I2F
	cvt.rn.f32.u32 rf1, ru1;
	cvt.rz.f32.u32 rf2, ru2;
	cvt.rm.f32.u32 rf3, ru3;
	cvt.rp.f32.u32 rf4, ru4;
	cvt.rn.f32.u64 rf5, rul5;
	cvt.rn.f64.s64 rfl6,rsl6;

//store. Without this step the computation above will be skipped by ptxas.
	st.f32 [ru0+0x0], rf1;
	st.f32 [ru0+0x4], rf2;
	st.f32 [ru0+0xc], rf3;
	st.f32 [ru0+0x10],rf4;
	st.f32 [ru0+0x14],rf5;
	st.f64 [ru0+0x18],rfl6;	

	exit;
}


```