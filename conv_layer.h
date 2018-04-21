#ifndef CNN_CONV_LAYER_H
#define CNN_CONV_LAYER_H

#include "halide_image_io.h"
#include "Halide.h"
#include <string>

using namespace Halide;
using namespace Halide::Tools;

class ConvLayer
{
public:
    std::string name;
    int input_dim, stride, sub_size;
    int *dim;

    Var x, y, c, f;
    Func mirrored, sub_sum, aver, err;
    Func pre_norm, conv;
    
    
    ConvLayer(Buffer<uint8_t>& in_layer, Buffer<float>& kernel, 
                Buffer<float>& bias, std::string _name, int _input_dim, int* _dim, 
                int kernel_size, int _stride, int shift) {
        printf("add another conv layer.\n");
        name = _name;
        input_dim = _input_dim;
        stride = _stride;
        dim = _dim;
        sub_size = input_dim/stride*input_dim/stride;

        mirrored = BoundaryConditions::mirror_image(in_layer);
        RDom filter(0, kernel_size, 0, kernel_size, 0, dim[2]);
        pre_norm(x, y, f) = 0.f;
        pre_norm(x, y, f) += mirrored(x*stride+filter.x-shift, 
                                y*stride+filter.y-shift, filter.z)
                                *kernel(filter.x, filter.y, filter.z, f);
    }

    ConvLayer(Buffer<float>& in_layer, Buffer<float>& kernel, 
                Buffer<float>& bias, std::string _name, int _input_dim, int* _dim, 
                int kernel_size, int _stride, int shift) {
        printf("add another conv layer.\n");
        name = _name;
        input_dim = _input_dim;
        stride = _stride;
        dim = _dim;
        sub_size = input_dim/stride*input_dim/stride;

        mirrored = BoundaryConditions::mirror_image(in_layer);
        RDom filter(0, kernel_size, 0, kernel_size, 0, dim[2]);
        pre_norm(x, y, f) = 0.f;
        pre_norm(x, y, f) += mirrored(x * stride+filter.x-shift, 
                                y * stride+filter.y-shift, filter.z)
                                * kernel(filter.x, filter.y, filter.z, f);


    }

    void addReLU() {
        RDom reLU(0, input_dim/stride, 0, input_dim/stride, 0, dim[3]);
        reLU.where(conv(reLU.x, reLU.y, reLU.z) < 0.f);
        conv(reLU.x, reLU.y, reLU.z) = 0.f;
    }
    void batchNorm(Buffer<float>& pre_activation, Buffer<float>& scale, 
            Buffer<float>& shift, float epsilon=1e-3) {
        
        RDom r(0, input_dim/stride, 0, input_dim/stride);
        sub_sum(f) = 0.f;
        sub_sum(f) += pre_activation(r.x, r.y, f);
        aver(f) = sub_sum(f)/sub_size;
        Buffer<float> tmp1 = aver.realize(dim[3]);

        err(f) = 0.f;
        err(f) += (pre_activation(r.x, r.y, f)-tmp1(f))
                    * (pre_activation(r.x, r.y, f)-tmp1(f));
        Buffer<float> tmp2 = err.realize(dim[3]);

        conv(x, y, f) = shift(f);
        conv(x, y, f) += scale(f)*(pre_activation(x, y, f) - tmp1(f))
                        /sqrt(tmp2(f)/sub_size+epsilon);
    }
};

#endif
