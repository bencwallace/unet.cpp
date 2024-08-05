#include <iostream>
#include <string>

#include "io.h"
#include "ops.h"
#include "unet.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " [input] [output]" << std::endl;
    return 1;
  }
  std::string in_path = argv[1];
  tensor<3> input = read_ppm(in_path);
  std::string out_path = argv[2];

  unet model(3, 2);
  model.load_checkpoint("weights.bin");
  input = scale(input, 1.0 / 255);
  tensor<3> logits = model(input);
  tensor<2> mask = argmax(logits);
  mask = scale(mask, 255);
  write_pgm(out_path, mask);
}
