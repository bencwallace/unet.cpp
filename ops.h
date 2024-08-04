#pragma once

#include <cmath>
#include <cstring>
#include <optional>

#include "tensor.hpp"

tensor<3> conv(const tensor<3> &input, const tensor<4> &kernel, const tensor<1> &bias, int padding = 0);

tensor<3> conv_transpose(const tensor<3> &input, const tensor<4> &kernel, const tensor<1> &bias);

tensor<3> max_pool(const tensor<3> &input, int size);

tensor<3> cat(const tensor<3> &a, const tensor<3> &b);

tensor<2> argmax(const tensor<3> &input);

template <int Rank> tensor<Rank> scale(const tensor<Rank> &input, float factor) {
  tensor<Rank> output(input.dims_);
  for (int i = 0; i < input.size_; ++i) {
    output.data_[i] = input.data_[i] * factor;
  }
  return output;
}

template <int Rank>
tensor<Rank> batch_norm(const tensor<Rank> &input, const tensor<1> &mean, const tensor<1> &variance,
                        const tensor<1> &scale, const tensor<1> &offset, float epsilon = 1e-5) {
  tensor<Rank> output(input.dims_);
  for (int c = 0; c < input.dims_[0]; ++c) {
    for (int y = 0; y < input.dims_[1]; ++y) {
      for (int x = 0; x < input.dims_[2]; ++x) {
        float in = input({c, y, x});
        float normalized = (in - mean({c})) / std::sqrt(variance({c}) + epsilon);
        output({c, y, x}) = scale({c}) * normalized + offset({c});
      }
    }
  }
  return output;
}

template <int Rank> tensor<Rank> relu(const tensor<Rank> &input) {
  tensor<Rank> output(input.dims_);
  for (int i = 0; i < input.size_; ++i) {
    output.data_[i] = std::max(0.0f, input.data_[i]);
  }
  return output;
}
