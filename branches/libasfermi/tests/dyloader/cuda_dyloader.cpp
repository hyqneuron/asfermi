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

#include "cuda.h"
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
#include <malloc.h>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

using namespace std;

// The maximum number of registers per thread.
#define MAX_REGCOUNT		63

// The register footprint of uberkern loader code itself.
// Engine still should be able to run dynamic kernels with
// smaller footprints, but when loader code is running, this
// number is a must.
#define LOADER_REGCOUNT		7

// Ad extra offset between the end of uberkerel loader code
// and the first dynamic kernel code
#define BASE_EXTRA_OFFSET	1024

// An extra offset between loaded dynamic kernels codes to
// force no caching/prefetching.
#define EXTRA_OFFSET		512

static int verbose = 1;

// Device memory buffer for binary size and content.
struct buffer_t
{
	// Dynamic kernel binary size.
	unsigned int szbinary;

	// Dynamic kernel binary to load (if not yet loaded).
	char* binary;
};

struct CUDYloader_t;

struct CUDYfunction_t
{
	unsigned int szbinary;
	vector<char> binary;
	
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
			ElfW(Ehdr)* elf_header = (ElfW(Ehdr)*)content.c_str();
			size_t size = (size_t)elf_header->e_phoff +
				elf_header->e_phentsize *  elf_header->e_phnum;
			e = elf_memory((char*)content.c_str(), size);
			size_t shstrndx;
			if (elf_getshdrstrndx(e, &shstrndx))
			{
				if (verbose)
					cerr << "Cannot get the CUBIN/ELF strings section header index" << endl;
				throw CUDA_ERROR_INVALID_SOURCE;
			}
			Elf_Scn* scn = NULL;
			while ((scn = elf_nextscn(e, scn)) != NULL)
			{
				// Get section name.
				GElf_Shdr shdr;
				if (gelf_getshdr(scn, &shdr) != &shdr)
				{
					if (verbose)
						cerr << "Cannot load the CUBIN/ELF section header" << endl;
					throw CUDA_ERROR_INVALID_SOURCE;
				}
				char* name = NULL;
				if ((name = elf_strptr(e, shstrndx, shdr.sh_name)) == NULL)
				{
					if (verbose)
						cerr << "Cannot load the CUBIN/ELF section name" << endl;
					throw CUDA_ERROR_INVALID_SOURCE;
				}

				if (elfname != name) continue;
		
				// Extract regcount out of 24 bits of section info.
				regcount = shdr.sh_info >> 24;

				// Extract binary opcodes and size.
				szbinary = shdr.sh_size;
				binary.resize(szbinary);
				memcpy(&binary[0], (char*)content.c_str() + shdr.sh_offset, szbinary);
				
				// For asynchronous data transfers to work, need to
				// pin memory for binary content.
				CUresult cuerr = cuMemHostRegister(&binary[0], szbinary, 0);
				if (cuerr != CUDA_SUCCESS)
				{
					if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
					{
						if (verbose)
							cerr << "Cannot pin memory for CUBIN binary content" << endl;		
						throw cuerr;
					}
					
					// We are fine, if memory is already registered.
				}

				// Also need to pin offset.
				cuerr = cuMemHostRegister(&offset, sizeof(unsigned int), 0);
				if (cuerr != CUDA_SUCCESS)
				{
					if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
					{
						if (verbose)
							cerr << "Cannot pin memory for the dynamic kernel pool offset" << endl;
						throw cuerr;
					}

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
		{
			if (verbose)
				cerr << "Cannot determine the kernel regcount" << endl;
			throw CUDA_ERROR_INVALID_SOURCE;
		}

		if (verbose)
			cout << "regcount = " << regcount << ", size = " << szbinary << endl;
	}

	CUDYfunction_t(CUDYloader_t* loader,
		char* opcodes, size_t nopcodes, int regcount) :
	
	loader(loader), szbinary(8 * nopcodes), regcount(regcount)
	
	{
		// Copy binary.
		binary.resize(szbinary);
		memcpy(&binary[0], opcodes, szbinary);
	
		// For asynchronous data transfers to work, need to
		// pin memory for binary content.
		CUresult cuerr = cuMemHostRegister(&binary[0], szbinary, 0);
		if (cuerr != CUDA_SUCCESS)
		{
			if (cuerr != CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED)
				throw cuerr;

			// We are fine, if memory is already registered.
		}
		
		// Also need to pin offset.
		cuerr = cuMemHostRegister(&offset, sizeof(unsigned int), 0);
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
		CUresult cuerr = cuMemHostUnregister(&binary[0]);
		if (cuerr != CUDA_SUCCESS)
		{
			if (cuerr != CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED)
				throw cuerr;
			
			// We are fine, if memory is already unregistered.
		}

		// Unpin pinned memory for offset.
		cuerr = cuMemHostUnregister(&offset);
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
	
	CUDYloader_t(int capacity) : offset(BASE_EXTRA_OFFSET), capacity(BASE_EXTRA_OFFSET + capacity * 8), buffer(0)
	{
		int ntokens = sizeof(uberkern) / sizeof(const char*);

		int device;
		CUresult curesult = cuDeviceGet(&device, 0);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot get the CUDA device" << endl;
			throw curesult;
		}
		int major = 2, minor = 0;
		curesult = cuDeviceComputeCapability(&major, &minor, device);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot get the CUDA device compute capability" << endl;
			throw curesult;
		}

		// Select the bank number: differs between Fermi and Kepler.
		string bank = "";
		if (major == 2) bank = "[0x2]";
		if (major == 3) bank = "[0x3]";

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

			// Replace $BANK with [0x2] or [0x3], depending on target architecture.
			for (size_t index = line.find("$BANK", 0);
				index = line.find("$BANK", index); index++)
			{
				if (index == string::npos) break;
				line.replace(index, bank.length(), bank);
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
			stream << "/* 0x0000 */\tJMP c" << bank << "[0x8];" << endl;
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
			// Emit cubin for the current device architecture.
			size_t size;
			cubin = asfermi_encode_cubin(csource, major * 10 + minor, 0, &size);
			if (!cubin)
			{
				if (verbose)
					cerr << "Cannot encode the uberkern into cubin" << endl;
				throw CUDA_ERROR_INVALID_SOURCE;
			}

			// Load binary containing uberkernel to deivce memory.
			curesult = cuModuleLoadData(&module, cubin);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot load uberkern module" << endl;
				throw curesult;
			}
			moduleLoaded = true;

			// Load uberkern loader entry point from module.
			curesult = cuModuleGetFunction(&loader, module, "uberkern");
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot load uberkern loader function" << endl;
				throw curesult;
			}

			// Load uberkernel entry points from module.
			for (int i = 0; i < MAX_REGCOUNT + 1; i++)
			{
				stringstream stream;
				stream << "uberkern" << i;
				string name = stream.str();
				curesult = cuModuleGetFunction(&entry[i], module, name.c_str());
				if (curesult != CUDA_SUCCESS)
				{
					if (verbose)
						cerr << "Cannot load uberkern entry function" << endl;
					throw curesult;
				}
			}

			// Load the uberkernel command constant.
			curesult = cuModuleGetGlobal(&command, NULL, module, "uberkern_cmd");
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot load the uberkern_cmd constant" << endl;
				throw curesult;
			}

			// Initialize command value with ZERO, so on the next
			// launch uberkern will simply report LEPC and exit.
			curesult = cuMemsetD32(command, 0, 1);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot write the uberkern_cmd constant" << endl;
				throw curesult;
			}

			// Load the dynamic kernel code BRA target address.
			curesult = cuModuleGetGlobal(&address, NULL, module, "uberkern_goto");
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot load the uberkern_goto constant" << endl;
				throw curesult;
			}

			// Load the uberkernel config structure address constant.
			curesult = cuModuleGetGlobal(&config, NULL, module, "uberkern_config");
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot write the uberkern_config constant" << endl;
				throw curesult;
			}

			// Allocate space for uberkernel arguments.
			curesult = cuMemAlloc(&buffer, sizeof(buffer_t) + this->capacity);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot allocate the uberkern memory buffer" << endl;
				throw curesult;
			}
			binary = (CUdeviceptr)((char*)buffer + sizeof(buffer_t));

			// Fill the structure address constant with the address value.
			curesult = cuMemcpyHtoD(config, &buffer, sizeof(CUdeviceptr*));
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot write the uberkern config structure address" << endl;
				throw curesult;
			}

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
			curesult = cuLaunchKernel(loader,
				1, 1, 1, 1, 1, 1, 0, 0, NULL, params);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot launch the uberkern loader" << endl;
				throw curesult;
			}

			// Synchronize kernel.
			curesult = cuCtxSynchronize();
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot synchronize the uberkern loader" << endl;
				throw curesult;
			}

			// Read the LEPC.
			curesult = cuMemcpyDtoH(&lepc, buffer, sizeof(int));
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot read the uberkern LEPC value" << endl;
				throw curesult;
			}

			if (verbose)
				cout << "LEPC = 0x" << hex << lepc << dec << endl;

			// Set binary pointer in buffer.
			curesult = cuMemcpyHtoD((CUdeviceptr)((char*)buffer + 8), &binary, sizeof(CUdeviceptr*));
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot write the uberkern binary pointer" << endl;
				throw curesult;
			}

			// Pin memory for offset.
			curesult = cuMemHostRegister(&offset, sizeof(int), 0);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot pin host memory for the offset" << endl;
				throw curesult;
			}
		}
		catch (CUresult cuerr)
		{
			if (cubin) free(cubin);
			if (moduleLoaded)
			{
				CUresult curesult = cuModuleUnload(module);
				if (curesult != CUDA_SUCCESS)
					if (verbose)
						cerr << "Cannot unload the uberkern module" << endl;
			}
			if (buffer)
			{
				CUresult curesult = cuMemFree(buffer);
				if (curesult != CUDA_SUCCESS)
					if (verbose)
						cerr << "Cannot free the uberkern memory buffer" << endl;
			}
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
	
		CUresult curesult = cuModuleUnload(module);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot unload the uberkern module" << endl;
			throw curesult;
		}
		curesult = cuMemFree(buffer);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot free the uberlern memory buffer" << endl;
			throw curesult;
		}
		
		// Unpin memory for offset.
		curesult = cuMemHostUnregister(&offset);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot unpin host memory for the offset" << endl;
			throw curesult;
		}
	}
	
	CUresult Load(CUDYfunction_t* function, CUstream stream)
	{
		// Check the dynamic pool has enough free space to
		// incorporate the specified dynamic kernel body.
		if (offset + function->szbinary > capacity)
		{
			if (verbose)
				cerr << "Insufficient space in the uberkern memory pool" << endl;
			throw CUDA_ERROR_OUT_OF_MEMORY;
		}

		// Set dynamic kernel binary size.
		CUresult curesult = cuMemcpyHtoDAsync(buffer,
			&function->szbinary, sizeof(unsigned int), stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot set the dynamic kernel binary size" << endl;
			throw curesult;
		}

		// Initialize command value with ONE, so on the next
		// launch uberkern will load dynamic kernel code and exit.
		curesult = cuMemsetD32Async(command, 1, 1, stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot write the uberkern config command" << endl;
			throw curesult;
		}

		// Fill the dynamic kernel code BRA target address.
		curesult = cuMemcpyHtoDAsync(address, &offset, sizeof(int), stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot fill the dynamic kernel code BRA target address" << endl;
			throw curesult;
		}

		// Load dynamic kernel binary.
		curesult = cuMemcpyHtoDAsync((CUdeviceptr)binary,
			&function->binary[0], function->szbinary, stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot load the dynamic kernel binary" << endl;
			throw curesult;
		}

		// Synchronize stream.
		curesult = cuStreamSynchronize(stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot synchronize after the dynamic kernel binary loading" << endl;
			throw curesult;
		}

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
		curesult = cuLaunchKernel(loader,
			1, 1, 1, 1, 1, 1, 0, stream, NULL, params);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot launch the uberkern loader" << endl;
			throw curesult;
		}

		// Synchronize stream.
		curesult = cuStreamSynchronize(stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot synchronize the uberkern loader" << endl;
			throw curesult;
		}

		// Store function body offset.
		function->offset = offset;

		// Increment pool offset by the size of kernel binary.
		offset += function->szbinary + EXTRA_OFFSET;
		
		// Track function for disposal.
		functions.push_back(function);

		return CUDA_SUCCESS;
	}
	
	CUresult Launch(CUDYfunction_t* function,
		unsigned int gx, unsigned int gy, unsigned int gz,
		unsigned int bx, unsigned int by, unsigned int bz,
		size_t szshmem, void* args, CUstream stream, float* time)
	{
		// Initialize command value with LEPC, so on the next
		// launch uberkern will load dynamic kernel code and exit.
		// XXX: 0x138 is #BRA of uberkernel loader code - the value
		// may change if loader code gets changed. 
		CUresult curesult = cuMemsetD32Async(command, lepc + 0x138, 1, stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot write the uberkern config command" << endl;
			return curesult;
		}

		// Fill the dynamic kernel code BRA target address.
		curesult = cuMemcpyHtoDAsync(address, &function->offset, sizeof(int), stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot fill the dynamic kernel code BRA target address" << endl;
			return curesult;
		}

		// Synchronize stream.
		curesult = cuStreamSynchronize(stream);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot synchronize after the dynamic kernel binary loading" << endl;
			return curesult;
		}

		CUevent start, stop;
		if (time)
		{
			curesult = cuEventCreate(&start, 0);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot create timer start event" << endl;
				return curesult;
			}
			curesult = cuEventCreate(&stop, 0);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot create timer stop event" << endl;
				return curesult;
			}
			curesult = cuEventRecord(start, stream);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot record the timer start event" << endl;
				return curesult;
			}
		}

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
		curesult = cuLaunchKernel(entry[function->regcount],
			gx, gy, gz, bx, by, bz, szshmem,
			stream, NULL, config);
		if (curesult != CUDA_SUCCESS)
		{
			if (verbose)
				cerr << "Cannot launch the dynamic kernel" << endl;
			return curesult;
		}

		if (time)
		{
			curesult = cuEventRecord(stop, stream);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot record the timer stop event" << endl;
				return curesult;
			}
			curesult = cuEventSynchronize(stop);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot synchronize the dynamic kernel" << endl;
				return curesult;
			}
			curesult = cuEventElapsedTime(time, start, stop);
			if (curesult != CUDA_SUCCESS)
			{
				if (verbose)
					cerr << "Cannot get the timer elapsed time" << endl;
				return curesult;
			}
			*time *= 1e-3;
		}

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
	size_t szshmem, void* args, CUstream stream, float* time)
{
	try
	{
		return function->loader->Launch(function,
			gx, gy, gz, bx, by, bz, szshmem, args, stream, time);
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

