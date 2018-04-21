#ifndef CNN_SETTING_H
#define CNN_SETTING_H
// Variable created inside namespace
namespace SettingCNN
{
    int WeightDims[16][4] = {
        // 00 is padding 40x40
        {9, 9, 3, 32},      // 01
        {3, 3, 32, 64},     // 02
        {3, 3, 64, 128},    // 03
        {3, 3, 128, 128},   // 04
        {3, 3, 128, 128},   // 05
        {3, 3, 128, 128},   // 06
        {3, 3, 128, 128},   // 07
        {3, 3, 128, 128},   // 08
        {3, 3, 128, 128},   // 09
        {3, 3, 128, 128},   // 10
        {3, 3, 128, 128},   // 11
        {3, 3, 128, 128},   // 12
        {3, 3, 128, 128},   // 13
        {3, 3, 128, 64},    // 14
        {3, 3, 64, 32},     // 15
        {9, 9, 32, 3},       // 16              
    };
    int StrideDims[16] = {
        1, 2, 2, 
        1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 
        -2, -2, 1
    };
    int** InputDims;
}

#endif