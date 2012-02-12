.PHONY: ../../libasfermi.so

all: libuberkern.so test1 test2 test3 test4 test5

../../libasfermi.so:
	cd ../.. && $(MAKE)

libuberkern.so: init.o launch.o dispose.o cuda_dyloader.o ../../libasfermi.so
	nvcc -shared $^ -o $@ -L../.. -lasfermi -lcuda -lelf

init.o: init.c loader.h
	nvcc -I../.. -Xcompiler -fPIC -Xcompiler -std=c99 -g -c $< -o $@

launch.o: launch.c
	nvcc -I../.. -Xcompiler -fPIC -g -c $< -o $@

dispose.o: dispose.c
	nvcc -Xcompiler -fPIC -g -c $< -o $@

cuda_dyloader.o: cuda_dyloader.cpp cuda_dyloader.h loader.h
	nvcc -I../.. -Xcompiler -fPIC -g -c $< -o $@

clean:
	rm -rf *.o libuberkern.so test1 test2
	cd ../.. && $(MAKE) clean

test1: test1.c libuberkern.so
	nvcc -g $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda

test2: test2.cpp libuberkern.so
	nvcc -g $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda

test3: test3.c libuberkern.so
	nvcc -g -Xcompiler -std=c99 -DBLOCK_SIZE=256 $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda

test4: test4.c libuberkern.so
	nvcc -I../.. -g -Xcompiler -std=c99 -DBLOCK_SIZE=256 $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda

test5: test5.cu libuberkern.so
	nvcc -arch=sm_20 -I. -g $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda 

tests: test1 test2 test3 test4 test5
	perl tests.pl

snap:
	tar -cvzf ../uberkern_`date +%y%m%d%H%M%S`.tar.gz ../uberkern

