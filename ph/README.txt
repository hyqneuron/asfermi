genmake.txt:
	the makefile used to generate makefile for the project
	Before doing a remake, please delete generated.txt first
	Contains a copy and a del command, which need to be changed into cp and rm for non-windows environment.

generated.txt: 
	the the generated makefile from make -f genmake
	use make -f generated.txt to build the project
	Contains a del command.

gentemp.txt:
	Template for the generation of generated.txt
	Contains a del command.

Before building the project it'd be good to run g++ stdafx.h first to generate the precompiled header.