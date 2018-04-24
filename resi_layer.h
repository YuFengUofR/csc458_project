#ifndef CNN_RESI_LAYER_H
#define CNN_RESI_LAYER_H

#include "halide_image_io.h"
#include "Halide.h"
#include "input_layer.h"
#include "setting.h"
#include "conv_layer.h"
#include <string>

using namespace Halide;
using namespace Halide::Tools;

class ResiLayer
{
public:
    std::string name;
    int input_dim, stride, shift, sub_size;
    int *dim;
    float epsilon=1e-3;

    Var x, y, c, f;
    Func pre_resi, resi, pre_conv, conv;
    Buffer<float> tmp1, tmp2;
    Buffer<float> tmp, pre_act1, pre_act2;

    ResiLayer(Buffer<float>& in_layer, InputLayer* input_layer, int id,
                std::string name, int* dim, int input_dim, 
                int kernel_size, int stride, int shift) {

        printf("add another resi layer.\n");
        ConvLayer* conv_1 = new ConvLayer(in_layer, input_layer->getWeightAt(id),
                            input_layer->getBiasAt(id), name + "conv_1",
                            input_dim, dim, kernel_size, stride, shift);

        Buffer<float> pre = conv_1->pre_norm.realize(input_dim, 
                                            input_dim, dim[3]);
        if (SettingCNN::use_gpu)
            pre.copy_to_host();
        conv_1->batchNorm(pre, input_layer->getScaleAt(id),
                            input_layer->getShiftAt(id));
        conv_1->addReLU();
        Buffer<float> ret = conv_1->conv.realize(input_dim, 
                                            input_dim, dim[3]);

        ConvLayer* conv_2 = new ConvLayer(ret, input_layer->getWeightAt(id+1),
                            input_layer->getBiasAt(id+1), name + "conv_2", 
                            input_dim, dim, kernel_size, stride, shift);
        
        pre = conv_2->pre_norm.realize(input_dim, input_dim, dim[3]);
        if (SettingCNN::use_gpu)
            pre.copy_to_host();
        conv_2->batchNorm(pre, input_layer->getScaleAt(id+1),
                            input_layer->getShiftAt(id+1));
        conv_2->addReLU();

        ret = conv_2->conv.realize(input_dim, input_dim, dim[3]);

        resi(x, y, f) = in_layer(x, y, f) + ret(x, y, f);
    }
};

#endif