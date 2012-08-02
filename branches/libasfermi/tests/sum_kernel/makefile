.PHONY: ../../libasfermi.so

all: sum_kernel

../../libasfermi.so:
	cd ../.. && $(MAKE) asfermi libasfermi.so

sum_kernel: sum_kernel.cu sum_kernel.cubin
	nvcc -g -DBLOCK_SIZE=256 $< -o $@ -lcuda

sum_kernel.cubin: sum_kernel.s ../../libasfermi.so
	../../asfermi $< -64 -o $@

clean:
	rm -rf sum_kernel sum_kernel.cubin
