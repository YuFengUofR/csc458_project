#ifndef CNN_TRANS_LAYER_H
#define CNN_TRANS_LAYER_H

#include "halide_image_io.h"
#include "Halide.h"
#include <string>

using namespace Halide;
using namespace Halide::Tools;

class TransLayer
{
public:
    std::string name;
    int input_dim, stride, sub_size;
    int *dim;

    Var x, y, c, f;
    Func pre_norm, sub_sum, aver, err;
    Func upsample, trans, mirrored1, mirrored2;
    
    TransLayer(Buffer<float>& in_layer, Buffer<float>& kernel, 
                Buffer<float>& bias, std::string _name, int* _dim, int _input_dim,
                int kernel_size, int _stride, int shift, bool add_relu) {
        
        printf("add another trans layer.\n");
        name = _name;
        input_dim = _input_dim;
        stride = _stride;
        dim = _dim;
        sub_size = input_dim*stride*input_dim*stride;

        mirrored1 = BoundaryConditions::mirror_image(in_layer);
        RDom up(0, input_dim*stride, 0, input_dim*stride, 0, dim[2]);
        upsample(x, y, c) = 0.0f;
        up.where(up.x % stride == 0);
        up.where(up.y % stride == 0);
        upsample(up.x, up.y, up.z) = mirrored1(up.x/2, up.y/2, up.z);
        Buffer<float> tmp = upsample.realize(input_dim*stride, input_dim*stride, dim[2]);

        mirrored2 = BoundaryConditions::mirror_image(tmp);

        RDom filter(0, kernel_size, 0, kernel_size, 0, dim[2]);
        pre_norm(x, y, f) = 0.f;
        pre_norm(x, y, f) += mirrored2(x+filter.x-shift, y+filter.y-shift, filter.z) 
                            * kernel(filter.x, filter.y, filter.z, f);
    }

    void addReLU() {
        RDom reLU(0, input_dim*stride, 0, input_dim*stride, 0, dim[3]);
        reLU.where(trans(reLU.x, reLU.y, reLU.z) < 0.f);
        trans(reLU.x, reLU.y, reLU.z) = 0.f;
    }
    void batchNorm(Buffer<float>& pre_activation, Buffer<float>& scale, 
            Buffer<float>& shift, float epsilon=1e-3) {
        RDom r(0, input_dim*stride, 0, input_dim*stride);

        sub_sum(f) = 0.f;
        sub_sum(f) += pre_activation(r.x, r.y, f);
        aver(f) = sub_sum(f)/sub_size;
        Buffer<float> tmp1 = aver.realize(dim[3]);

        err(f) = 0.f;
        err(f) += (pre_activation(r.x, r.y, f)-tmp1(f))
                    * (pre_activation(r.x, r.y, f)-tmp1(f));
        Buffer<float> tmp2 = err.realize(dim[3]);
        
        trans(x, y, f) = shift(f);
        trans(x, y, f) += scale(f)*(pre_activation(x, y, f) - tmp1(f))
                        /sqrt(tmp2(f)/sub_size+epsilon);
    }
};

#endif