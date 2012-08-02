.PHONY: ../../libasfermi.so

all: constant3

../../libasfermi.so:
	cd ../.. && $(MAKE) asfermi libasfermi.so

constant3: constant3.cu constant3.cubin
	nvcc -g $< -o $@ -lcuda

constant3.cubin: constant3.s ../../libasfermi.so
	../../asfermi $< -sm_30 -64 -o $@

clean:
	rm -rf constant3 constant3.cubin
