#!/usr/bin/perl -w

my($path) = "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:.:../..";
my($cmd) = "";

$cmd = "$path ./test1 100";
print "$cmd\n"; system("$cmd");

$cmd = "$path ./test2 1000 10";
print "$cmd\n"; system("$cmd");

my($i);
for ($i = 256; $i < 32768; $i += 256)
{
	$cmd = "$path ./test3 $i";
	print "$cmd\n"; system("$cmd");
}

