#ifndef CNN_TRANS_LAYER_H
#define CNN_TRANS_LAYER_H

#include "halide_image_io.h"
#include "Halide.h"
#include "setting.h"
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
    Var x_o, x_i, x_i_o, x_i_i;
    Var y_o, y_i, c_i, c_o, f_o, f_i, tile_index;
    Func pre_norm, aver, err;
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
        RDom up(0, input_dim*stride, 0, input_dim*stride);
        upsample(x, y, c) = 0.0f;
        up.where(up.x % stride == 0 && up.y % stride == 0);

        upsample(up.x, up.y, c) = mirrored1(up.x/2, up.y/2, c);

        upsample.parallel(c);
        Buffer<float> tmp = upsample.realize(input_dim*stride, input_dim*stride, dim[2]);
        printf("done upsample image...\n");
        mirrored2 = BoundaryConditions::mirror_image(tmp);

        RDom filter(0, kernel_size, 0, kernel_size, 0, dim[2]);
        pre_norm(x, y, f) = 0.f;
        pre_norm(x, y, f) += mirrored2(x+filter.x-shift, y+filter.y-shift, filter.z) 
                            * kernel(filter.x, filter.y, filter.z, f);

        // // scheduling...
        if (SettingCNN::use_gpu) {
            pre_norm.update()
                    .reorder(x, y, filter.z);
            pre_norm.compute_root();
            pre_norm.update()
                    .reorder(x, y, filter.z);
            pre_norm
                    .vectorize(x, 8)
                    .split(y, y, y_i, 32)                 
                    .update()
                    .split(f, f, f_i, 8);

            pre_norm
                    .update()
                    .unroll(filter.x)
                    .unroll(filter.y)
                    .fuse(y, f, tile_index);
            pre_norm
                    .update()       
                    .gpu_blocks(tile_index)
                    .gpu_threads(x);

            mirrored2.compute_at(pre_norm, filter.z);

            Target target = get_host_target();
            if (target.os == Target::OSX) {
                target.set_feature(Target::Metal);
            } else {
                target.set_feature(Target::OpenCL);
            }
            pre_norm.compile_jit(target);
            
        } 
        
    }

    void addReLU() {
        RDom reLU(0, input_dim*stride, 0, input_dim*stride, 0, dim[3]);
        reLU.where(trans(reLU.x, reLU.y, reLU.z) < 0.f);
        trans(reLU.x, reLU.y, reLU.z) = 0.f;
    }
    void batchNorm(Buffer<float>& pre_activation, Buffer<float>& scale, 
            Buffer<float>& shift, float epsilon=1e-3) {
        RDom r(0, input_dim*stride, 0, input_dim);

        aver(f) = 0.f;
        aver(f) += pre_activation(r.x, r.y*2, f) + pre_activation(r.x, r.y*2+1, f);
        aver(f) = aver(f)/sub_size;

        if (SettingCNN::use_gpu) {
            aver.update()
                .split(f, f_o, f_i, 1)
                .gpu_blocks(f_o)
                .gpu_threads(f_i);

            Target target = get_host_target();
            if (target.os == Target::OSX) {
                target.set_feature(Target::Metal);
            } else {
                target.set_feature(Target::OpenCL);
            }
            aver.compile_jit(target);   
        }
        
        Buffer<float> tmp1 = aver.realize(dim[3]);
        if (SettingCNN::use_gpu) tmp1.copy_to_host();

        err(f) = 0.f;
        err(f) += (pre_activation(r.x, r.y, f)-tmp1(f))
                    * (pre_activation(r.x, r.y, f)-tmp1(f));

        err(f) = err(f)/sub_size;
        if (SettingCNN::use_gpu) {
            err
            .split(f, f_o, f_i, 1)
            .gpu_blocks(f_o)
            .gpu_threads(f_i);

            Target target = get_host_target();
            if (target.os == Target::OSX) {
                target.set_feature(Target::Metal);
            } else {
                target.set_feature(Target::OpenCL);
            }
            err.compile_jit(target);   
        }
        Buffer<float> tmp2 = err.realize(dim[3]);
        if (SettingCNN::use_gpu) tmp2.copy_to_host();

        trans(x, y, f) = shift(f);
        trans(x, y, f) += scale(f)*(pre_activation(x, y, f) - tmp1(f))
                        /sqrt(tmp2(f)/sub_size+epsilon);
    }
};

#endif