#include "unet.h"
#include "ops.h"

void double_conv::load_checkpoint(std::ifstream &file) {
  for (int i = 0; i < 2; i++) {
    load_weights(file, conv_weights_[i]);
    load_weights(file, bn_scales_[i]);
    load_weights(file, bn_offsets_[i]);
    load_weights(file, bn_means_[i]);
    load_weights(file, bn_vars_[i]);
  }
}

tensor<3> double_conv::operator()(const tensor<3> &input) {
  tensor<3> output = conv(input, conv_weights_[0], conv_biases[0], 1);
  output = batch_norm(output, bn_means_[0], bn_vars_[0], bn_scales_[0], bn_offsets_[0]);
  output = relu(output);

  output = conv(output, conv_weights_[1], conv_biases[1], 1);
  output = batch_norm(output, bn_means_[1], bn_vars_[1], bn_scales_[1], bn_offsets_[1]);
  output = relu(output);

  return output;
}

void down::load_checkpoint(std::ifstream &file) { dconv.load_checkpoint(file); }

tensor<3> down::operator()(const tensor<3> &input) {
  tensor<3> pooled = max_pool(input, 2);
  tensor<3> output = dconv(pooled);
  return output;
}

void up::load_checkpoint(std::ifstream &file) {
  load_weights(file, conv_transpose_weights_);
  load_weights(file, conv_transpose_biases_);
  dconv_.load_checkpoint(file);
}

tensor<3> up::operator()(const tensor<3> &input, const tensor<3> &skip) {
  tensor<3> output = conv_transpose(input, conv_transpose_weights_, conv_transpose_biases_);
  // TODO: pad output to match skip (only necessary if image dimensions are not powers of 2)
  output = cat(skip, output);
  output = dconv_(output);
  return output;
}

void unet::load_checkpoint(const std::string &checkpoint) {
  std::ifstream file(checkpoint, std::ios::binary);
  if (!file) {
    std::cerr << "Failed to open file" << std::endl;
    std::exit(1);
  }

  inc.load_checkpoint(file);
  down1.load_checkpoint(file);
  down2.load_checkpoint(file);
  down3.load_checkpoint(file);
  down4.load_checkpoint(file);
  up1.load_checkpoint(file);
  up2.load_checkpoint(file);
  up3.load_checkpoint(file);
  up4.load_checkpoint(file);
  load_weights(file, conv_weights_);
  load_weights(file, conv_biases_);
}

tensor<3> unet::operator()(const tensor<3> &input) {
  tensor<3> output1 = inc(input);
  tensor<3> output2 = down1(output1);
  tensor<3> output3 = down2(output2);
  tensor<3> output4 = down3(output3);
  tensor<3> output = down4(output4);
  output = up1(output, output4);
  output = up2(output, output3);
  output = up3(output, output2);
  output = up4(output, output1);
  output = conv(output, conv_weights_, conv_biases_);
  return output;
}
