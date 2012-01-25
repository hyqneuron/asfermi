.PHONY: ../../libasfermi.so

all: libuberkern.so test1 test2

../../libasfermi.so:
	cd ../.. && $(MAKE)

libuberkern.so: init.o launch.o dispose.o ../../libasfermi.so
	nvcc -shared init.o launch.o dispose.o -o $@ -L../.. -lasfermi -lcuda

init.o: init.c
	nvcc -I../.. -Xcompiler -fPIC -Xcompiler -std=c99 -g -c $< -o $@

launch.o: launch.c
	nvcc -I../.. -Xcompiler -fPIC -g -c $< -o $@

dispose.o: dispose.c
	nvcc -Xcompiler -fPIC -g -c $< -o $@

clean:
	rm -rf *.o libuberkern.so test1 test2
	cd ../.. && $(MAKE) clean

test1: test1.c libuberkern.so
	nvcc -g $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda

test2: test2.cpp libuberkern.so
	nvcc -g $< -o $@ -L. -luberkern -L../.. -lasfermi -lcuda

tests: test1 test2
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:.:../.. ./test1 100
	LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:.:../.. ./test2 1000 10

snap:
	tar -cvzf ../uberkern_`date +%y%m%d%H%M%S`.tar.gz ../uberkern

