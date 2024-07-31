#pragma once

#include <fstream>

#include "io.h"
#include "tensor.hpp"

struct double_conv {

  tensor<4> conv_weights_[2];
  tensor<1> conv_biases[2];
  tensor<1> bn_means_[2];
  tensor<1> bn_vars_[2];
  tensor<1> bn_scales_[2];
  tensor<1> bn_offsets_[2];

  double_conv(int in_channels, int out_channels)
      : conv_weights_{tensor<4>({out_channels, in_channels, 3, 3}), tensor<4>({out_channels, out_channels, 3, 3})},
        conv_biases{tensor<1>({out_channels}, 0), tensor<1>({out_channels}, 0)},
        bn_means_{tensor<1>({out_channels}), tensor<1>({out_channels})}, bn_vars_{tensor<1>({out_channels}),
                                                                                  tensor<1>({out_channels})},
        bn_scales_{tensor<1>({out_channels}), tensor<1>({out_channels})}, bn_offsets_{tensor<1>({out_channels}),
                                                                                      tensor<1>({out_channels})} {}

  void load_checkpoint(std::ifstream &file);

  tensor<3> operator()(const tensor<3> &input);
};

struct down {

  double_conv dconv;

  down(int in_channels, int out_channels) : dconv(in_channels, out_channels) {}

  void load_checkpoint(std::ifstream &file);

  tensor<3> operator()(const tensor<3> &input);
};

struct up {

  tensor<4> conv_transpose_weights_;
  tensor<1> conv_transpose_biases_;
  double_conv dconv_;

  up(int in_channels, int out_channels)
      : conv_transpose_weights_({in_channels, in_channels / 2, 2, 2}), conv_transpose_biases_({in_channels / 2}),
        dconv_(in_channels, out_channels) {}

  void load_checkpoint(std::ifstream &file);

  tensor<3> operator()(const tensor<3> &input, const tensor<3> &skip);
};

struct unet {

  double_conv inc;
  down down1;
  down down2;
  down down3;
  down down4;
  up up1;
  up up2;
  up up3;
  up up4;
  tensor<4> conv_weights_;
  tensor<1> conv_biases_;

  unet(int channels, int classes)
      : inc(channels, 64), down1(64, 128), down2(128, 256), down3(256, 512), down4(512, 1024), up1(1024, 512),
        up2(512, 256), up3(256, 128), up4(128, 64), conv_weights_({classes, 64, 1, 1}), conv_biases_({64}) {}

  void load_checkpoint(const std::string &checkpoint);

  tensor<3> operator()(const tensor<3> &input);
};
