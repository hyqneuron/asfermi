.PHONY: ../../libasfermi.so

all: libdyloader.so test1 test2 test3 test4 test5

../../libasfermi.so:
	cd ../.. && $(MAKE)

libdyloader.so: cuda_dyloader.o ../../libasfermi.so
	nvcc -shared $^ -o $@ -L../.. -lasfermi -lcuda -lelf

cuda_dyloader.o: cuda_dyloader.cpp cuda_dyloader.h loader.h
	nvcc -I../.. -Xcompiler -fPIC -g -c $< -o $@

clean:
	rm -rf *.o libdyloader.so test1 test2 test3 test4 test5
	cd ../.. && $(MAKE) clean

test1: test1.c libdyloader.so
	nvcc -g $< -o $@ -L. -ldyloader -L../.. -lasfermi -lcuda

test2: test2.cpp libdyloader.so
	nvcc -g $< -o $@ -L. -ldyloader -L../.. -lasfermi -lcuda

test3: test3.c libdyloader.so
	nvcc -g -Xcompiler -std=c99 -DBLOCK_SIZE=256 $< -o $@ -L. -ldyloader -L../.. -lasfermi -lcuda

test4: test4.c libdyloader.so
	nvcc -I../.. -g -Xcompiler -std=c99 -DBLOCK_SIZE=256 $< -o $@ -L. -ldyloader -L../.. -lasfermi -lcuda

test5: test5.cu libdyloader.so
	nvcc -arch=sm_20 -I. -g $< -o $@ -L. -ldyloader -L../.. -lasfermi -lcuda 

tests: test1 test2 test3 test4 test5
	perl tests.pl

snap:
	tar -cvzf ../dyloader_`date +%y%m%d%H%M%S`.tar.gz ../dyloader

