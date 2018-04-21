// the actual complie command is:
// c++ main.cpp -g -std=c++11 -I ../include -I ../tools -L ../bin -lHalide `libpng-config --cflags --ldflags` -ljpeg -o main

// The only Halide header file you need is Halide.h. It includes all of Halide.
// Include some support code for loading pngs.
#include "halide_image_io.h"
#include "Halide.h"
#include "conv_layer.h"
#include "trans_layer.h"
#include "max_pool.h"
#include "input_layer.h"
#include "resi_layer.h"
#include "setting.h"
#include <sys/time.h>
#include <string>
#include <vector>

using namespace Halide;
using namespace Halide::Tools;
using namespace SettingCNN;


void computeInputDims(int w, int h, int c);
void computeDimAt(int i, int w, int h, int c);

int main(int argc, char **argv) {

    if (argc != 2) {
        printf("ERROR::please enter the image path.\n");
    }
    std::string img_name = argv[1];

    // This program defines a single-stage imaging pipeline that
    // brightens an image.
    struct timespec start, end;

    InputDims = (int **) calloc(17, sizeof(int *));
    for (int i = 0; i < 17; i++) {
        InputDims[i] = (int *) calloc(3, sizeof(int));
    }

    InputLayer* input_layer = new InputLayer(img_name);
    Buffer<uint8_t>& image = input_layer->getInput();
    int width = image.width();
    int height = image.height();
    // int height = 64;
    if (width < height) {
        height = width;
    } else {
        width = height;
    }
    int channels = image.channels();
    computeInputDims(width, height, channels);

    for (int i = 0; i < 16; i++) {
        std::string w_fn = "data/w_" + std::to_string(i+1) + ".out"; 
        input_layer->addWeight(w_fn, WeightDims[i][0], 
                                WeightDims[i][2], WeightDims[i][3]);
        std::string sh_fn = "data/shift_" + std::to_string(i+1) + ".out";
        input_layer->addShift(sh_fn, WeightDims[i][3]);
        std::string sc_fn = "data/scale_" + std::to_string(i+1) + ".out"; 
        input_layer->addScale(sc_fn, WeightDims[i][3]);
        input_layer->addBias("B", WeightDims[i][3]);
    }
    printf("loading weights and biases done!\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    ConvLayer* conv_1 = new ConvLayer(image, 
                                        input_layer->getWeightAt(0),
                                        input_layer->getBiasAt(0), 
                                        "conv_1", InputDims[0][0],
                                        input_layer->getDimAt(0)
                                        , 9, 1, 4);
    Buffer<float> pre_1_1 = conv_1->pre_norm.realize(InputDims[1][0], 
                                InputDims[1][1], InputDims[1][2]);
    conv_1->batchNorm(pre_1_1, input_layer->getScaleAt(0),
                                input_layer->getShiftAt(0));
    conv_1->addReLU();
    Buffer<float> ret_1_1 = conv_1->conv.realize(InputDims[1][0], 
                                InputDims[1][1], InputDims[1][2]);
    printf("No. %d conv layer computation done!\n", 1);

    ConvLayer* conv_2 = new ConvLayer(ret_1_1, 
                                        input_layer->getWeightAt(1),
                                        input_layer->getBiasAt(1), 
                                        "conv_2", InputDims[1][0],
                                        input_layer->getDimAt(1)
                                        , 3, 2, 1);
    Buffer<float> pre_1_2 = conv_2->pre_norm.realize(InputDims[2][0], 
                                InputDims[2][1], InputDims[2][2]);

    conv_2->batchNorm(pre_1_2, input_layer->getScaleAt(1),
                                input_layer->getShiftAt(1));
    conv_2->addReLU();

    Buffer<float> ret_1_2 = conv_2->conv.realize(InputDims[2][0], 
                                InputDims[2][1], InputDims[2][2]);
    printf("No. %d conv layer computation done!\n", 2);

    ConvLayer* conv_3 = new ConvLayer(ret_1_2, 
                                        input_layer->getWeightAt(2),
                                        input_layer->getBiasAt(2), 
                                        "conv_3", InputDims[2][0],
                                        input_layer->getDimAt(2)
                                        , 3, 2, 1);
    Buffer<float> pre_1_3 = conv_3->pre_norm.realize(InputDims[3][0], 
                                InputDims[3][1], InputDims[3][2]);

    conv_3->batchNorm(pre_1_3, input_layer->getScaleAt(2),
                                input_layer->getShiftAt(2));

    conv_3->addReLU();
    Buffer<float> ret_1_3 = conv_3->conv.realize(InputDims[3][0], 
                                InputDims[3][1], InputDims[3][2]);
    printf("No. %d conv layer computation done!\n", 3);

    ResiLayer* resi_1 = new ResiLayer(ret_1_3, input_layer, 3, "resi_1", 
                    input_layer->getDimAt(3), InputDims[3][0], 3, 1, 1);

    Buffer<float> ret_2_1 = resi_1->resi.realize(InputDims[5][0], 
                                InputDims[5][1], InputDims[5][2]);

    printf("No. %d resi layer computation done!\n", 1);

    ResiLayer* resi_2 = new ResiLayer(ret_2_1, input_layer, 5, "resi_2", 
                    input_layer->getDimAt(5), InputDims[5][0], 3, 1, 1);

    Buffer<float> ret_2_2 = resi_2->resi.realize(InputDims[7][0], 
                                InputDims[7][1], InputDims[7][2]);

    printf("No. %d resi layer computation done!\n", 2);

    ResiLayer* resi_3 = new ResiLayer(ret_2_2, input_layer, 7, "resi_3", 
                    input_layer->getDimAt(7), InputDims[7][0], 3, 1, 1);

    Buffer<float> ret_2_3 = resi_3->resi.realize(InputDims[9][0], 
                                InputDims[9][1], InputDims[9][2]);

    printf("No. %d resi layer computation done!\n", 3);

    ResiLayer* resi_4 = new ResiLayer(ret_2_3, input_layer, 9, "resi_4", 
                    input_layer->getDimAt(9), InputDims[9][0], 3, 1, 1);

    Buffer<float> ret_2_4 = resi_4->resi.realize(InputDims[11][0], 
                                InputDims[11][1], InputDims[11][2]);

    printf("No. %d resi layer computation done!\n", 4);

    ResiLayer* resi_5 = new ResiLayer(ret_2_4, input_layer, 11, "resi_5", 
                    input_layer->getDimAt(11), InputDims[11][0], 3, 1, 1);

    Buffer<float> ret_2_5 = resi_5->resi.realize(InputDims[13][0], 
                                InputDims[13][1], InputDims[13][2]);

    printf("No. %d resi layer computation done!\n", 5);

    TransLayer* trans_1 = new TransLayer(ret_2_5, input_layer->getWeightAt(13), 
                input_layer->getBiasAt(13),  "trans_1", input_layer->getDimAt(13), 
                InputDims[13][0], 3, 2, 1, true);

    Buffer<float> pre_3_1 = trans_1->pre_norm.realize(InputDims[14][0], 
                                InputDims[14][1], InputDims[14][2]);
    trans_1->batchNorm(pre_3_1, input_layer->getScaleAt(13),
                                input_layer->getShiftAt(13));
    trans_1->addReLU();

    Buffer<float> ret_3_1 = trans_1->trans.realize(InputDims[14][0], 
                                InputDims[14][1], InputDims[14][2]);

    printf("No. %d trans layer computation done!\n", 1);

    TransLayer* trans_2 = new TransLayer(ret_3_1, input_layer->getWeightAt(14), 
                input_layer->getBiasAt(14),  "trans_2", input_layer->getDimAt(14), 
                InputDims[14][0], 3, 2, 1, true);

    Buffer<float> pre_3_2 = trans_2->pre_norm.realize(InputDims[15][0], 
                                InputDims[15][1], InputDims[15][2]);
    trans_2->batchNorm(pre_3_2, input_layer->getScaleAt(14),
                                input_layer->getShiftAt(14));
    trans_2->addReLU();

    Buffer<float> ret_3_2 = trans_2->trans.realize(InputDims[15][0], 
                                InputDims[15][1], InputDims[15][2]);

    printf("No. %d trans layer computation done!\n", 2);

    ConvLayer* final = new ConvLayer(ret_3_2, input_layer->getWeightAt(15),
                                        input_layer->getBiasAt(15), 
                                        "final", InputDims[15][0],
                                        input_layer->getDimAt(15)
                                        , 9, 1, 4);
    
    Buffer<float> pre_result = final->pre_norm.realize(InputDims[16][0], 
                                InputDims[16][1], InputDims[16][2]);
    final->batchNorm(pre_result, input_layer->getScaleAt(15),
                                input_layer->getShiftAt(15));


    printf("Final layer computation done!\n");

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t delta_ms = (end.tv_sec-start.tv_sec) * 1000 
                        + (end.tv_nsec-start.tv_nsec) / 1000000;
    printf("Total run time : %llu ms.\n", delta_ms);

    Func recast;
    Var x, y, c;
    recast(x, y, c) = Halide::cast<uint8_t>(
        max(0, min(tanh(final->conv(x, y, c))*150.f+128.f, 255)));

    Halide::Buffer<uint8_t> output =
        recast.realize(InputDims[16][0], InputDims[16][1], InputDims[16][2]);

    Tools::save_image(output, "strange.png");

    printf("Success!\n");
    return 0;
}

void computeInputDims(int w, int h, int c) {
    InputDims[0][0] = w;
    InputDims[0][1] = h;
    InputDims[0][2] = c;
    for (int i = 1; i < 17; i++) {
        computeDimAt(i, InputDims[i-1][0], InputDims[i-1][1],
                    InputDims[i-1][2]);
    }
}

void computeDimAt(int i, int w, int h, int c) {
    if (StrideDims[i-1] > 0) {
        InputDims[i][0] = w/StrideDims[i-1];
        InputDims[i][1] = h/StrideDims[i-1];
        InputDims[i][2] = WeightDims[i-1][3];
    } else {
        InputDims[i][0] = -w*StrideDims[i-1];
        InputDims[i][1] = -h*StrideDims[i-1];
        InputDims[i][2] = WeightDims[i-1][3];
    }
}