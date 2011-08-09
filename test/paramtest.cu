extern "C"{
__global__ void kernel(int *addr, int *addr2, int val3)
{
	addr[0] = 0;
	addr[1] = 1;
	addr[2] = 2;
	addr2[0] = val3;
	addr2[1] = val3;
	addr2[2] = val3;
}

__global__ void kernel2(int *addr, int *addr2, int val3)
{
	addr[0] = 0;
	addr[1] = 1;
	addr[2] = 2;
	addr2[0] = val3;
	addr2[1] = val3;
	addr2[2] = val3;
}

}