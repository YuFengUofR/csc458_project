#ifndef CNN_INPUT_LAYER_H
#define CNN_INPUT_LAYER_H
#include "halide_image_io.h"
#include "Halide.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

using namespace Halide;
using namespace Halide::Tools;

class InputLayer
{
public:
    Buffer<uint8_t> image;
    std::vector<Buffer<float>> kernels;
    std::vector<Buffer<float>> biases; 
    std::vector<Buffer<float>> scales;
    std::vector<Buffer<float>> shifts; 
    std::vector<int *> dims;

    InputLayer(std::string filename) {
        srand (static_cast <unsigned> (time(0)));
        image = load_image(filename);
        printf("Loading image succeed..\n");
    };
    void addWeight(std::string filename, int f_size, int f_c, int f_num) {
        std::vector<float> nums;
        std::string line;
        std::ifstream fid;
        fid.open(filename);
        // fid.open(,(filename.c_str());
        if (fid.is_open())
        {
            while (getline(fid,line))
            {
                nums.push_back(stof(line));
            }
        }
        fid.close();
        int index;
        Buffer<float> kernel(f_size, f_size, f_c, f_num);
        for (int n=0; n<f_num; n++) {
            for (int k=0; k<f_c; k++) {
                for (int j=0; j<f_size; j++) {
                    for (int i=0; i<f_size; i++) {
                        index = i + j*f_size + k*f_size*f_size
                                    + n*f_size*f_size*f_c; 
                        kernel(i, j, k, n) = nums.at(index);
                    }
                }
            }
        }

        int *dim = (int *) calloc(4, sizeof(int)); 
        dim[0] = f_size;
        dim[1] = f_size;
        dim[2] = f_c;
        dim[3] = f_num; 
        kernels.push_back(kernel);
        dims.push_back(dim);
        // printf("add weight \'%s\'\n", filename.c_str());

    }
    void addScale(std::string filename, int f_num) {
        std::vector<float> nums;
        std::string line;
        std::ifstream fid;
        fid.open(filename);
        // fid.open(,(filename.c_str());
        if (fid.is_open())
        {
            while (getline(fid,line))
            {
                nums.push_back(stof(line));
            }
        }
        fid.close();
        Buffer<float> scale(f_num);
        for (int n=0; n<f_num; n++) {
            scale(n) = nums.at(n);
        }
        scales.push_back(scale);
    }
    void addShift(std::string filename, int f_num) {
        std::vector<float> nums;
        std::string line;
        std::ifstream fid;
        fid.open(filename);
        // fid.open(,(filename.c_str());
        if (fid.is_open())
        {
            while (getline(fid,line))
            {
                nums.push_back(stof(line));
            }
        }
        fid.close();
        Buffer<float> shift(f_num);
        for (int n=0; n<f_num; n++) {
            shift(n) = nums.at(n);
        }
        shifts.push_back(shift);
    }
    void addBias(std::string filename, int f_num) {
        Buffer<float> bias(f_num);
        biases.push_back(bias);
        // printf("add bias \'%s\'\n", filename.c_str());
    }
    Buffer<float> &getWeightAt(int num) {
        printf("get weight layer:\'%d\'\n", num);
        return kernels.at(num);
    }
    Buffer<float> &getBiasAt(int num) {
        printf("get bias layer:\'%d\'\n", num);
        return biases.at(num);
    }
    Buffer<float> &getScaleAt(int num) {
        printf("get scale layer:\'%d\'\n", num);
        return scales.at(num);
    }
    Buffer<float> &getShiftAt(int num) {
        printf("get shift layer:\'%d\'\n", num);
        return shifts.at(num);
    }
    Buffer<uint8_t> &getInput() {
        return image;
    }
    int *getDimAt(int num) {
        printf("get dim layer:\'%d\'\n", num);
        return dims.at(num);
    }
};

#endif
