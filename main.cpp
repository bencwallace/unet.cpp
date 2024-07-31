#include "io.h"
#include "ops.h"
#include "unet.h"

int main() {
  unet model(3, 2);
  model.load_checkpoint("weights.bin");
  tensor<3> input = read_ppm("car.ppm");
  input = scale(input, 1.0 / 255);
  tensor<3> logits = model(input);
  tensor<2> mask = argmax(logits);
  mask = scale(mask, 255);
  write_pgm("mask.pgm", mask);
}
