#ifndef CNN_RESI_LAYER_H
#define CNN_RESI_LAYER_H

#include "halide_image_io.h"
#include "Halide.h"
#include "input_layer.h"
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
        conv_1->batchNorm(pre, input_layer->getScaleAt(id),
                            input_layer->getShiftAt(id));
        conv_1->addReLU();
        Buffer<float> ret = conv_1->conv.realize(input_dim, 
                                            input_dim, dim[3]);

        ConvLayer* conv_2 = new ConvLayer(ret, input_layer->getWeightAt(id+1),
                            input_layer->getBiasAt(id+1), name + "conv_2", 
                            input_dim, dim, kernel_size, stride, shift);

        pre = conv_2->pre_norm.realize(input_dim, input_dim, dim[3]);
        conv_2->batchNorm(pre, input_layer->getScaleAt(id+1),
                            input_layer->getShiftAt(id+1));
        conv_2->addReLU();

        ret = conv_2->conv.realize(input_dim, input_dim, dim[3]);

        resi(x, y, f) = in_layer(x, y, f) + ret(x, y, f);
    }

    
    // ResiLayer(Buffer<float>& in_layer, Buffer<float>& kernel1, 
    //             Buffer<float>& bias1, Buffer<float>& kernel2, 
    //             Buffer<float>& bias2, std::string _name, int* _dim,
    //             int _input_dim, int kernel_size, int _stride, int _shift) {
    //     printf("add another resi layer.\n");
    //     name = _name;
    //     input_dim = _input_dim;
    //     stride = _stride;
    //     shift = _shift;
    //     dim = _dim;
    //     sub_size = input_dim/stride*input_dim/stride;
    //     RDom filter(0, kernel_size, 0, kernel_size, 0, dim[2]);

    //     pre_conv(x, y, f) = bias1(f);
    //     pre_conv(x, y, f) += in_layer(x*stride+filter.x, 
    //                                 y*stride+filter.y, filter.z) *
    //                                 kernel1(filter.x, filter.y, filter.z, f);

    //     pre_act1 = pre_conv.realize(input_dim-2*shift, input_dim-2*shift, dim[3]);
    //     batchNorm(pre_act1, 2);

    //     conv(x, y, f) = (pre_act1(x, y, f) - tmp1(f))/(tmp2(f)-epsilon);

    //     addReLU();

    //     tmp = conv.realize(input_dim-2*shift, input_dim-2*shift, dim[3]);
    
    //     pre_resi(x, y, f) = bias2(f) + in_layer(x+2*shift, y+2*shift, f);
    //     pre_resi(x, y, f) += tmp(x*stride+filter.x, y*stride+filter.y, filter.z) *
    //                             kernel2(filter.x, filter.y, filter.z, f);

    //     pre_act2 = pre_resi.realize(input_dim-4*shift, input_dim-4*shift, dim[3]);
    //     batchNorm(pre_act2, 4);

    //     resi(x, y, f) = (pre_act2(x, y, f) - tmp1(f))/(tmp2(f)-epsilon);
    // }
    

    // void addReLU() {
    //     RDom reLU(0, input_dim-2*shift, 0, input_dim-2*shift, 0, dim[3]);
    //     reLU.where(conv(reLU.x, reLU.y, reLU.z) < 0.f);
    //     conv(reLU.x, reLU.y, reLU.z) = 0.f;
    // }

    // void batchNorm(Buffer<float>& pre_activation, int cnt) {
    //     RDom r(0, input_dim-cnt*shift, 0, input_dim-cnt*shift);
    //     Func aver, err;
    //     aver(f) = sum(pre_activation(r.x, r.y, f))/sub_size;
    //     tmp1 = aver.realize(dim[3]);

    //     err(f) = sum(abs(pre_activation(r.x, r.y, f)-tmp1(f)))/sub_size;
    //     tmp2 = err.realize(dim[3]);
    // }
};

#endif