# Utilities #
Note for cuobjdump: When invalid opcode is present in the cubin, the hex digits output by cuobjdump will be wrong(get shifted). However, the hex number length and disassembled instruction information will still be correct.

---

## cuobjdump output processor ##
Recovers correct opcode binary digits from the output of cuobjdump.
Usage:
```
cuobjdump -sass(or something else) file.cubin>in.txt
cuprocess in.txt out.txt
```

The program makes heavy assumption about the input file and produces messy stuff for non-instruction lines, though that should not matter as long as you follow the usage above.

[Source](http://code.google.com/p/asfermi/source/browse/trunk/utilities/cuprocess.cpp)


---

## cubin binary editor ##
A very simple binary mode editor I made for myself. Written in C#.NET. Contains binary for Windows with .NET Framework 3.5 or higher.

This program displays opcode in binary mode and the least significant bit is displayed first.

#### Usage ####
```
CUBINEditor [inputfile [offset]]
```

Or, just open program. Click "Browse", choose file, then click "Load".

  1. Click "File info" after loading the cubin to locate the kernel section. Look for .text.kernelname.
  1. Scroll to the instruction location you want.
  1. If your instruction location has an offset that is not a multiple of 8, enter (offset%8) into the num box and click "Set Offset". Do editing as you normally would with a binary editor.
  1. Click "Write Back" to store the file to the location specified in the textbox. You can just change the text in the textbox before saving.

Ctrl+Up = go to the box above

Ctrl+Down = go to box below

**Note**:
  1. You will need [Mono](http://www.go-mono.com/mono-downloads/download.html) if your OS does not already support .NET 3.5.
  1. May contain bug, as most of it was written while I was half-asleep. This program is written with minimal complexity so you could expect crashes whenever you do something I did not expect.
[Download](http://code.google.com/p/asfermi/downloads/detail?name=cubinEditor%20v5.zip)

#### Change log ####
  1. v2: Added support for command-line argument.
  1. v3: Added a button "File Info". Other 3 disabled buttons are just there for my own use. The "Fill bits" button permutes na and nb bits for 512 instructions. If you want to use this button please modify its property, Enabled, as well as the clicking event handling code on your own. Currently it has some display glitches when running on Mono under Ubuntu, and the FileInfo page doesn't show up properly. I shouldn't have used the ListView structure.
  1. v4: The display problems are fixed. Not sure how well it runs on native Linux but on my VirtualBox Ubuntu it was terribly slow. Note that this package libmono-winforms2.0-cil is probably needed for Mono to run correctly.
  1. v5: Added support for 64-bit files.