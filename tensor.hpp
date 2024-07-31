#pragma once

#include <array>
#include <memory>
#include <numeric>

template <int Rank> struct tensor {

  int size_;
  std::array<int, Rank> dims_;
  std::array<int, Rank> strides_;
  std::unique_ptr<float[]> data_;

  tensor(const std::array<int, Rank> &dims)
      : size_(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<int>())), dims_(dims),
        data_(std::make_unique<float[]>(size_)) {
    strides_[Rank - 1] = 1;
    for (int i = Rank - 2; i >= 0; --i) {
      strides_[i] = strides_[i + 1] * dims_[i + 1];
    }
  }

  tensor(const std::array<int, Rank> &dims, float value) : tensor(dims) {
    std::fill(data_.get(), data_.get() + size_, value);
  }

  float &operator()(const std::array<int, Rank> &indices) {
    int index = std::inner_product(indices.begin(), indices.end(), strides_.begin(), 0);
    return data_[index];
  }
  float operator()(const std::array<int, Rank> &indices) const {
    return const_cast<tensor *>(this)->operator()(indices);
  }
};