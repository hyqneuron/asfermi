.PHONY: ../../libasfermi.so

all: constant2

../../libasfermi.so:
	cd ../.. && $(MAKE) asfermi libasfermi.so

constant2: constant2.cu constant2.cubin
	nvcc -g $< -o $@ -lcuda

constant2.cubin: constant2.s ../../libasfermi.so
	../../asfermi $< -64 -o $@

clean:
	rm -rf constant2 constant2.cubin
