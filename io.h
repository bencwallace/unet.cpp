#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "tensor.hpp"

template <int Rank> void load_weights(std::ifstream &file, tensor<Rank> &tensor) {
  try {
    file.read(reinterpret_cast<char *>(tensor.data_.get()), tensor.size_ * sizeof(float));
  } catch (const std::exception &e) {
    std::cerr << "Failed to read file" << std::endl;
    std::exit(1);
  }
}

tensor<3> read_ppm(const std::string &filename);

void write_pgm(const std::string &filename, const tensor<2> &image);
