//Directly written on google code because I'm not yet in the mood of proper coding. This filoe is not supposed to go through compilation

#include <iostream>
#include <fstream>
#include <string>

//----- Basic Type Definitions -----
struct Parser
{
    char* Name;
    virtual void Parse() = 0;
};
struct LineParser:: Parser //need keyword public?
{
    LineParser(){Name = "DefaultLineParser"}
    void Parse()
    {
    }
};
struct InstructionParser:: Parser
{
    Iparser(){Name = "default_iparser"}
    int Function(Instruction instruction)
    {
    }
};
//----- End of Basic Type Definitions -----

//----- Global Variables -----
//----- End of Global Variables -----