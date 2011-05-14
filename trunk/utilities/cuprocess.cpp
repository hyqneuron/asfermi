#include <fstream>
#include <string>
using namespace std;

int check(int narg, char**args, fstream *input, fstream *output);
int process(fstream *input, fstream* output);
char* decode(char c1, char c2, char* result);

int main(int narg, char **args)
{
	fstream input, output;
	int result = check(narg, args, &input, &output);
	if(result!=0) return result;
	process(&input, &output);

	input.close();
	output.flush();
	output.close();
	puts("Finished");
	return 0;
}
//	/*0018*/     /*0xfc0f5de21bffffff*/ 	MOV32I R61, -0x1;

int process(fstream *input, fstream* output)
{
	char line[300];
	char* result = new char[9];
	result[8]=0;
	while(input->good())
	{
		input->getline(line, 300);
		string str(line);
		str+="		";
		for(int i=24; i>=18; i-=2)
			str+=decode(str[i], str[i+1], result);
		for(int i=32; i>=26; i-=2)
			str+=decode(str[i], str[i+1], result);
		str+="\n";
		output->write(str.data(), str.length());
	}
	delete result;

}

#define reduce(o)\
{			\
	if(o<58)	\
		o-=48;	\
	else o-=87;	\
}
static char code[]={0b00000001, 0b00000010, 0b00000100, 0b00001000};

char* decode(char c1, char c2, char* result)
{
	reduce(c1);
	reduce(c2);
	for(int i=0; i<4; i++)
	{
		result[i+4] = (int)(code[i]&c1)==0? '0':'1';
		result[i] = (int)(code[i]&c2)==0? '0':'1';
	}
	return result;
}


int check(int narg, char **args, fstream *input, fstream *output)
{
	puts("Version 0.3");
	if(narg<3)
	{
		puts("Usage: process inputfile outputfile");
		return -1;
	}
	input->open(args[1], fstream::in);
	if(!input->is_open())
	{
		puts("failed to open input file");
		return -2;
	}
	output->open(args[2], fstream::out);
	if(!output->is_open())
	{
		puts("failed to open output file");
		input->close();
		return -3;
	}
	return 0;
}