/*
 * Copyright (c) 2012 by Dmitry Mikushin
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "cuda_dyloader.h"
#include "libasfermi.h"
#include "loader.h"

#include <cstring>
#include <elf.h>
#include <fstream>
#include <gelf.h>
#include <iomanip>
#include <iostream>
#include <libelf.h>
#include <link.h>
#include <list>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

// The maximum number of registers per thread.
#define MAX_REGCOUNT	63

// The register footprint of uberkern loader code itself.
// Engine still should be able to run dynamic kernels with
// smaller footprints, but when loader code is running, this
// number is a must.
#define LOADER_REGCOUNT	7

#define CUTHROW(stmt) \
{ \
	CUresult result; \
	result = stmt; \
	if (result != CUDA_SUCCESS) throw result; \
}

// Device memory buffer for binary size and content.
struct buffer_t
{
	// Dynamic kernel binary size.
	unsigned int szbinary;

	// Dynamic kernel binary to load (if not yet loaded).
	char* binary;
};

struct CUDYloader_t;

// Get the ELF binary size, implementation by awalk
// https://bbs.archlinux.org/viewtopic.php?id=15298
static size_t elf_size(ElfW(Ehdr) *ehdr)
{
	// Find the first program header.
	ElfW(Phdr)* phdr = (ElfW(Phdr)*)((ElfW(Addr))ehdr + ehdr->e_phoff);

	// Find the final PT_LOAD segment's extent.
	ElfW(Addr) end;
	for (int i = 0; i < ehdr->e_phnum; i++)
	        if (phdr[i].p_type == PT_LOAD)
        		end = phdr[i].p_vaddr + phdr[i].p_memsz;

	// The start (virtual) address is always zero, so just return end.
	return (size_t)end;
}

struct CUDYfunction_t
{
	unsigned int szbinary;
	char* binary;
	
	short regcount;

	CUDYloader_t* loader;
	
	unsigned int offset;

	CUDYfunction_t(CUDYloader_t* loader,
		char* cubin, const char* name) : loader(loader), regcount(-1)
	{
		// Build kernel name as it should appear in cubin ELF.
		stringstream namestream;
		namestream << ".text." << name;
		string elfname = namestream.str();
	
		// Search for ELF magic in cubin. If found, then content
		// is supplied, otherwise - filename.
		stringstream stream(stringstream::in | stringstream::out |
			stringstream::binary);
		if (strncmp(cubin, ELFMAG, 4))
		{
			ifstream f(cubin, ios::in | ios::binary);
			stream << f.rdbuf();
			f.close();
		}
		else
		{
			stream << cubin;
		}
	
		// Extract kernel details: regcount, opcodes and their size.
		// Method: walk thorough the cubin using ELF tools and dump
		// details of the first kernel found (section starting with
		// ".text.").
		string content = stream.str();
		Elf* e = NULL;
		try
		{
			size_t size = elf_size((ElfW(Ehdr)*)content.c_str());
			e = elf_memory((char*)content.c_str(), size);
			size_t shstrndx;
			if (elf_getshdrstrndx(e, &shstrndx))
				throw CUDA_ERROR_INVALID_SOURCE;
			Elf_Scn* scn = NULL;
			while ((scn = elf_nextscn(e, scn)) != NULL)
			{
				scn = elf_nextscn(e, scn);

				// Get section name.
				GElf_Shdr shdr;
				if (gelf_getshdr(scn, &shdr) != &shdr)
					throw CUDA_ERROR_INVALID_SOURCE;
				char* name = NULL;
				if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
					throw CUDA_ERROR_INVALID_SOURCE;

				if (elfname != name) continue;
		
				// Extract regcount out of 24 bits of section info.
				regcount = shdr.sh_info >> 24;

				// Extract binary opcodes and size.
				binary = (char*)content.c_str() + shdr.sh_offset;
				szbinary = shdr.sh_size;
				
				// For asynchronous data transfers to work, need to
				// pin memory for binary content.
				CUresult cuerr = cuMemHostRegister(binary, szbinary, 0);
				if (cuerr != CUDA_SUCCESS)
				{
					if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
						throw cuerr;
					
					// We are fine, if memory is already registered.
				}

				break;
			}
		}
		catch (CUresult cuerr)
		{
			if (e) elf_end(e);
			throw cuerr;
		}
		elf_end(e);
	
		if (regcount == -1)
			throw CUDA_ERROR_INVALID_SOURCE;
	
		cout << "regcount = " << regcount << endl;
	}

	CUDYfunction_t(CUDYloader_t* loader,
		char* opcodes, size_t nopcodes, int regcount) :
	
	loader(loader), binary(opcodes), szbinary(8 * nopcodes),
	regcount(regcount)
	
	{
		// For asynchronous data transfers to work, need to
		// pin memory for binary content.
		CUresult cuerr = cuMemHostRegister(binary, szbinary, 0);
		if (cuerr != CUDA_SUCCESS)
		{
			if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
				throw cuerr;

			// We are fine, if memory is already registered.
		}
	}
	
	~CUDYfunction_t()
	{
		// Unpin pinned memory for binary.
		CUresult cuerr = cuMemHostUnregister(binary);
		if (cuerr != CUDA_SUCCESS)
		{
			if (cuerr != CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
				throw cuerr;
			
			// We are fine, if memory is already unregistered.
		}
	}
};

struct CUDYloader_t
{
	int offset;
	int capacity;
	
	CUmodule module;
	CUfunction loader, entry[MAX_REGCOUNT + 1];
	CUdeviceptr command, address, config, buffer, binary;
	int lepc;
	
	list<CUDYfunction_t*> functions;
	
	CUDYloader_t(int capacity) : offset(0), capacity(capacity * 8), buffer(0)
	{
		int ntokens = sizeof(uberkern) / sizeof(const char*);

		stringstream stream;
		stream << setfill('0');
		for (int i = 0, ilines = 0; i < ntokens; i++)
		{
			string line = uberkern[i];
			
			if (line[0] == '!')
			{
				stream << "\t\t" << line << endl;
				continue;
			}

			// Output the specified number of NOPs in place of $BUF.
			if (line == "$BUF")
			{
				for (unsigned int j = 0; j < capacity; j++)
				{
					stream << "/* 0x" << setw(4) << ilines * 8 << " */\tNOP;" << endl;
					ilines++;
				}
				continue;		
			}
		
			stream << "/* 0x" << setw(4) << ilines * 8 << " */\t" << line << ";" << endl;
			ilines++;
		}

		// Output additional entry points with all possible
		// reg counts.
		for (unsigned int regcount = 0; regcount < MAX_REGCOUNT + 1; regcount++)
		{
			stream << "\t\t!Kernel uberkern" << regcount << endl;
			stream << "\t\t!RegCount " << regcount << endl;
			stream << "\t\t!Param 256 1" << endl;
			stream << "/* 0x0000 */\tJMP c[0x2][0x8];" << endl;
			stream << "\t\t!EndKernel" << endl;
		}

		string source = stream.str();
		//cout << source;

		// Duplicate source array.
		std::vector<char> vsource(source.size() + 1);
		char* csource = (char*)&vsource[0];
		csource[source.size()] = '\0';
		memcpy(csource, source.c_str(), source.size());
		
		char* cubin = NULL;
		bool moduleLoaded = false;

		try
		{
			// Emit cubin.
			cubin = asfermi_encode_cubin(csource, 20, 0, NULL);
			if (!cubin) throw CUDA_ERROR_INVALID_SOURCE;

			// Load binary containing uberkernel to deivce memory.
			CUTHROW( cuModuleLoadData(&module, cubin) );
			moduleLoaded = true;

			// Load uberkern loader entry point from module.
			CUTHROW( cuModuleGetFunction(&loader, module, "uberkern") );
	
			// Load uberkernel entry points from module.
			for (int i = 0; i < MAX_REGCOUNT + 1; i++)
			{
				stringstream stream;
				stream << "uberkern" << i;
				string name = stream.str();
				CUTHROW( cuModuleGetFunction(&entry[i], module, name.c_str()) );
			}

			// Load the uberkernel command constant.
			CUTHROW( cuModuleGetGlobal(&command, NULL, module, "uberkern_cmd") );

			// Initialize command value with ZERO, so on the next
			// launch uberkern will simply report LEPC and exit.
			CUTHROW( cuMemsetD32(command, 0, 1) );

			// Load the dynamic kernel code BRA target address.
			CUTHROW( cuModuleGetGlobal(&address, NULL, module, "uberkern_goto") );

			// Load the uberkernel config structure address constant.
			CUTHROW( cuModuleGetGlobal(&config, NULL, module, "uberkern_config") );

			// Allocate space for uberkernel arguments.
			CUTHROW( cuMemAlloc(&buffer, sizeof(buffer_t) + this->capacity) );
			binary = buffer + sizeof(buffer_t);

			// Fill the structure address constant with the address value.
			CUTHROW( cuMemcpyHtoD(config, &buffer, sizeof(CUdeviceptr*)) );

			// Launch uberkernel to fill the LEPC.
			// Note we are always sending 256 Bytes, regardless
			// the actual size of arguments.
			char args[256];
			size_t szargs = 256;
			void* params[] =
			{
				CU_LAUNCH_PARAM_BUFFER_POINTER, args,
				CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
				CU_LAUNCH_PARAM_END
			};
			CUTHROW( cuLaunchKernel(loader,
				1, 1, 1, 1, 1, 1, 0, 0, NULL, params) );

			// Synchronize kernel.
			CUTHROW( cuCtxSynchronize() );

			// Read the LEPC.
			CUTHROW( cuMemcpyDtoH(&lepc, buffer, sizeof(int)) );

			cout << "LEPC = 0x" << hex << lepc << dec << endl;

			// Set binary pointer in buffer.
			CUTHROW( cuMemcpyHtoD(buffer + 8, &binary, sizeof(CUdeviceptr*)) );

			// Pin memory for offset.
			CUTHROW( cuMemHostRegister(&offset, sizeof(int), 0) );
		}
		catch (CUresult cuerr)
		{
			if (cubin) free(cubin);
			if (moduleLoaded) CUTHROW ( cuModuleUnload(module) );
			if (buffer) CUTHROW( cuMemFree(buffer) );
			throw cuerr;
		}
		free(cubin);
	}
	
	~CUDYloader_t()
	{
		// Dispose functions.
		for (list<CUDYfunction_t*>::iterator i = functions.begin(),
			ie = functions.end(); i != ie; i++)
			delete *i;
	
		CUTHROW ( cuModuleUnload(module) );
		CUTHROW ( cuMemFree(buffer) );
		
		// Unpin memory for offset.
		CUTHROW( cuMemHostUnregister(&offset) );
	}
	
	CUresult Load(CUDYfunction_t* function, CUstream stream)
	{
		// Check the dynamic pool has enough free space to
		// incorporate the specified dynamic kernel body.
		if (offset + function->szbinary > capacity)
			throw CUDA_ERROR_OUT_OF_MEMORY;

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );
	
		// Load dynamic kernel binary.
		CUTHROW( cuMemcpyHtoDAsync((CUdeviceptr)binary,
			function->binary, function->szbinary, stream) );

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );

		// Set dynamic kernel binary size.
		CUTHROW( cuMemcpyHtoDAsync(buffer,
			&function->szbinary, sizeof(unsigned int), stream) );

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );

		// Initialize command value with ONE, so on the next
		// launch uberkern will load dynamic kernel code and exit.
		CUTHROW( cuMemsetD32Async(command, 1, 1, stream) );

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );

		// Fill the dynamic kernel code BRA target address.
		CUTHROW( cuMemcpyHtoDAsync(address, &offset, sizeof(int), stream) );

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );

		// Launch uberkernel to load the dynamic kernel code.
		// Note we are always sending 256 Bytes, regardless
		// the actual size of arguments.
		char args[256];
		size_t szargs = 256;
		void* params[] =
		{
			CU_LAUNCH_PARAM_BUFFER_POINTER, args,
			CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
			CU_LAUNCH_PARAM_END
		};
		CUTHROW( cuLaunchKernel(loader,
			1, 1, 1, 1, 1, 1, 0, stream, NULL, params) );

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );

		// Store function body offset.
		function->offset = offset;

		// Increment pool offset by the size of kernel binary.
		offset += function->szbinary;
		
		// Track function for disposal.
		functions.push_back(function);

		return CUDA_SUCCESS;
	}
	
	CUresult Launch(CUDYfunction_t* function,
		unsigned int gx, unsigned int gy, unsigned int gz,
		unsigned int bx, unsigned int by, unsigned int bz,
		size_t szshmem, void* args, CUstream stream)
	{
		// Initialize command value with LEPC, so on the next
		// launch uberkern will load dynamic kernel code and exit.
		// XXX: 0x138 is #BRA of uberkernel loader code - the value
		// may change if loader code gets changed. 
		CUTHROW( cuMemsetD32Async(command, lepc + 0x138, 1, stream) );

		// Synchronize stream.
		CUTHROW( cuStreamSynchronize(stream) );

		// Launch device function.
		// Note we are always sending 256 Bytes, regardless
		// the actual size of arguments.
		size_t szargs = 256;
		void* config[] =
		{
			CU_LAUNCH_PARAM_BUFFER_POINTER, args,
			CU_LAUNCH_PARAM_BUFFER_SIZE, &szargs,
			CU_LAUNCH_PARAM_END
		};
		CUTHROW( cuLaunchKernel(entry[function->regcount],
			gx, gy, gz, bx, by, bz, szshmem,
			stream, NULL, config) );

		return CUDA_SUCCESS;
	}
};

// Initialize a new instance of CUDA dynamic loader with the
// specified capacity (in 8-byte instructions) in GPU memory.
CUresult cudyInit(CUDYloader* loader, int capacity)
{
	try
	{
		*loader = new CUDYloader_t(capacity);
	}
	catch (CUresult cuerr)
	{
		return cuerr;
	}
	return CUDA_SUCCESS;
}

// Load kernel function with the specified name from cubin file
// or memory buffer into dynamic loader context.
CUresult cudyLoadCubin(CUDYfunction* function,
	CUDYloader loader, char* cubin, const char* name,
	CUstream stream)
{
	try
	{
		// Create function.
		*function = new CUDYfunction_t(loader, cubin, name);
		return loader->Load(*function, stream);
	}
	catch (CUresult cuerr)
	{
		return cuerr;
	}
	return CUDA_SUCCESS;
}

// Load kernel function from the specified assembly opcodes
// into dynamic loader context.
CUresult cudyLoadOpcodes(CUDYfunction* function,
	CUDYloader loader, char* opcodes, size_t nopcodes,
	int regcount, CUstream stream)
{
	try
	{
		// Create function.
		*function = new CUDYfunction_t(loader, opcodes, nopcodes, regcount);
		return loader->Load(*function, stream);
	}
	catch (CUresult cuerr)
	{
		return cuerr;
	}
	return CUDA_SUCCESS;
}

// Launch kernel function through the dynamic loader.
CUresult cudyLaunch(CUDYfunction function,
	unsigned int gx, unsigned int gy, unsigned int gz,
	unsigned int bx, unsigned int by, unsigned int bz,
	size_t szshmem, void* args, CUstream stream)
{
	try
	{
		return function->loader->Launch(function,
			gx, gy, gz, bx, by, bz, szshmem, args, stream);
	}
	catch (CUresult cuerr)
	{
		return cuerr;
	}
}

// Dispose the specified CUDA dynamic loader instance.
CUresult cudyDispose(CUDYloader loader)
{
	delete loader;
	return CUDA_SUCCESS;
}

