#ifndef CNN_CONV_LAYER_H
#define CNN_CONV_LAYER_H

#include "halide_image_io.h"
#include "Halide.h"
#include "setting.h"
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
    Var x_o, x_i, y_o, y_i, c_i, c_o, f_o, f_i, tile_index;
    Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
    Func mirrored, aver, err;
    Func pre_norm, pre_norm_c, conv;
    
    
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
        pre_norm(x, y, f) += mirrored(x * stride+filter.x-shift, 
                                y * stride+filter.y-shift, filter.z)
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
                    .fuse(y, f, tile_index)
                    .parallel(tile_index);
            pre_norm
                    .update()       
                    .gpu_blocks(tile_index)
                    .gpu_threads(x);
            mirrored.compute_at(pre_norm, filter.z);

            Target target = get_host_target();
            if (target.os == Target::OSX) {
                target.set_feature(Target::Metal);
            } else {
                target.set_feature(Target::OpenCL);
            }
            pre_norm.compile_jit(target);
            
        } 

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

        // // scheduling...
        if (SettingCNN::use_gpu) {
            pre_norm.update()
                    .reorder(x, y, filter.z);
            pre_norm.compute_root();
            pre_norm.update()
                    .reorder(x, y, filter.z);

            pre_norm
                    .vectorize(x, 8)
                    .split(y, y, y_i, 32);    
            if (dim[3] >= 8) {
                pre_norm.update()
                    .split(f, f, f_i, 8);
            }

            pre_norm
                    .update()
                    .unroll(filter.x)
                    .unroll(filter.y)
                    .fuse(y, f, tile_index)
                    // .parallel(tile_index)
                    ;
            pre_norm
                    .update()       
                    .gpu_blocks(tile_index)
                    .gpu_threads(x);

            mirrored.compute_at(pre_norm, filter.z);

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
        RDom reLU(0, input_dim/stride, 0, input_dim/stride, 0, dim[3]);
        reLU.where(conv(reLU.x, reLU.y, reLU.z) < 0.f);
        conv(reLU.x, reLU.y, reLU.z) = 0.f;

        if (SettingCNN::use_gpu) {
            conv
                .split(x, x_o, x_i, 4)
                .vectorize(x_i)
                .gpu_blocks(f)
                .gpu_threads(y);

            Target target = get_host_target();
            if (target.os == Target::OSX) {
                target.set_feature(Target::Metal);
            } else {
                target.set_feature(Target::OpenCL);
            }
            conv.compile_jit(target);
            
        }

    }
    void batchNorm(Buffer<float>& pre_activation, Buffer<float>& scale, 
            Buffer<float>& shift, float epsilon=1e-3) {
        RDom r(0, input_dim/stride, 0, input_dim/stride);
        aver(f) = 0.f;
        aver(f) += pre_activation(r.x, r.y, f);
        aver(f) = aver(f)/sub_size;

        if (SettingCNN::use_gpu) {
            aver
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

        conv(x, y, f) = shift(f);
        conv(x, y, f) += scale(f)*(pre_activation(x, y, f) - tmp1(f))
                        /sqrt(tmp2(f)/sub_size+epsilon);

        
    }
};

#endif
