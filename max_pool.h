#include "halide_image_io.h"
#include "Halide.h"
#include <string>

using namespace Halide;
using namespace Halide::Tools;

class MaxPool
{
public:
    std::string name;
    Func pool;

    MaxPool(Func conv, std::string _name, int size) {
        Var x("x"), y("y"), f("f");
        name = _name;
        printf("add another max pool layer.\n");
        RDom r (0, size, 0, size);
        pool(x, y, f) = maximum(conv(x*2 + r.x, y*2 + r.y, f));        
    };
};