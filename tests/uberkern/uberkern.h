#ifndef UBERKERN_H
#define UBERKERN_H

#include <cuda.h>

#ifdef __cplusplus
extern "C"
{
#endif

// All kernels we need to pack into uberkernel have unified
// prototype: kernel(int*). So, for now let's just make the
// following prototype for uberkernel containing them:
// uberkernel(int*, struct uberkernel_args_t*)
struct uberkern_args_t
{
	// Dynamic kernel binary size.
	unsigned int szbinary;

	// Dynamic kernel binary to load (if not yet loaded).
	char* binary;
};

struct uberkern_entry_t
{
	// The base address of entry, relative to
	// the uberkernel starting point.
	size_t base;
	
	// Next entry in list.
	struct uberkern_entry_t* next;
};

struct uberkern_t
{
	// Uberkernel module.
	CUmodule module;

	// Uberkernel entry point (if loaded).
	CUfunction function;

	// Uberkernel binary (ELF cubin).
	char* binary;

	// Uberkernel register footprint.
	int regcount;
	
	// Uberkern opcodes count.
	int nopcodes;
	
	// The offset of the top most free space for
	// dynamically loaded kernels from the beginning of
	// pool (increments with every new kernel load).
	unsigned int offset;
	
	// The pool capacity (in bytes).
	unsigned int capacity;
	
	// Uberkernel arguments container in device
	// memory.
	struct uberkern_args_t* args;
	
	// Kernels loaded into uberkernel.
	struct uberkern_entry_t* entries;
	size_t nentries;
};

// Initialize a new instance of uberkernel with the
// specified capacity (in 8-byte instructions) in GPU memory.
// NOTE: this call assumes CUDA initialization and
// device selection is already performed on user side.
struct uberkern_t* uberkern_init(unsigned int capacity);

// Dispose the uberkernel handle, assuming it owns
// every non-NULL data field.
void uberkern_dispose(struct uberkern_t* uberkern);

// Launch entry in existing uberkernel. If entry is
// NULL, then load entry from the specified array
// of binary opcodes.
struct uberkern_entry_t* uberkern_launch(
	struct uberkern_t* uberkern, struct uberkern_entry_t* entry,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	size_t szshmem, void* args, char* binary, size_t szbinary);

// Unload existing entry from the uberkernel identified
// by handle.
void uberkern_unload(
	struct uberkern_t* uberkern,
	struct uberkern_entry_t* entry);

#ifdef __cplusplus
}
#endif

#endif // UBERKERN_H

