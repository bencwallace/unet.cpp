#include <fstream>
#include <sstream>

#include "io.h"

tensor<3> read_ppm(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  std::string line;
  std::getline(file, line);
  if (line != "P6") {
    throw std::runtime_error("Unsupported image format. Header: " + line);
  }

  std::stringstream ss;
  ss << file.rdbuf();

  int width = 0;
  int height = 0;
  int bit_depth = 0;
  ss >> width >> height;
  ss >> bit_depth;
  ss.ignore(1);

  tensor<3> image({3, height, width});
  unsigned char x;
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      ss.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
      image({0, h, w}) = static_cast<float>(x);

      ss.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
      image({1, h, w}) = static_cast<float>(x);

      ss.read(reinterpret_cast<char *>(&x), sizeof(unsigned char));
      image({2, h, w}) = static_cast<float>(x);
    }
  }

  return image;
}

void write_pgm(const std::string &filename, const tensor<2> &image) {
  std::ofstream file(filename, std::ios::binary | std::ios::out | std::ios::trunc);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  size_t height = image.dims_[0];
  size_t width = image.dims_[1];

  file << "P5\n";
  file << width << " " << height << '\n';
  file << "255\n";
  for (int i = 0; i < image.size_; ++i) {
    auto val = image.data_[i];
    auto temp = static_cast<unsigned char>(val);
    file.write(reinterpret_cast<const char *>(&temp), sizeof(char));
  }
}
