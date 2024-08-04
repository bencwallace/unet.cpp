#include <cassert>
#include <fstream>

#include "io.h"
#include "ops.h"
#include "unet.h"

const float epsilon = 1e-4;

// TODO: test all conv methods
void test_conv1() {
  tensor<3> input({2, 3, 3}, 1);
  tensor<4> kernel({1, 2, 3, 3}, 1);
  tensor<1> bias({3}, 0);
  tensor<3> output = conv(input, kernel, bias);

  assert(output.dims_[0] == 1);
  assert(output.dims_[1] == 1);
  assert(output.dims_[2] == 1);
  assert(output({0, 0, 0}) == 18);
}

void test_conv2() {
  tensor<3> input({2, 3, 3}, 1);
  tensor<4> kernel({1, 2, 3, 3}, 1);
  tensor<1> bias({3}, 0);
  tensor<3> output = conv(input, kernel, bias, 1);

  assert(output.dims_[0] == 1);
  assert(output.dims_[1] == 3);
  assert(output.dims_[2] == 3);

  assert(output({0, 0, 0}) == 8);
  assert(output({0, 0, 1}) == 12);
  assert(output({0, 0, 2}) == 8);
  assert(output({0, 1, 0}) == 12);
  assert(output({0, 1, 1}) == 18);
  assert(output({0, 1, 2}) == 12);
  assert(output({0, 2, 0}) == 8);
  assert(output({0, 2, 1}) == 12);
  assert(output({0, 2, 2}) == 8);
}

void test_conv3() {
  tensor<3> input({1, 3, 3});
  for (int h = 0; h < input.dims_[1]; ++h) {
    for (int w = 0; w < input.dims_[2]; ++w) {
      input({0, h, w}) = h * input.dims_[2] + w;
    }
  }
  tensor<4> kernel({2, 1, 2, 2}, 1);
  for (int h = 0; h < kernel.dims_[2]; ++h) {
    for (int w = 0; w < kernel.dims_[3]; ++w) {
      kernel({1, 0, h, w}) *= 2;
    }
  }
  tensor<1> bias({2}, -1);
  tensor<3> output = conv(input, kernel, bias);

  assert(output.dims_[0] == 2);
  assert(output.dims_[1] == 2);
  assert(output.dims_[2] == 2);

  assert(output({0, 0, 0}) == 7);
  assert(output({0, 0, 1}) == 11);
  assert(output({0, 1, 0}) == 19);
  assert(output({0, 1, 1}) == 23);

  assert(output({1, 0, 0}) == 15);
  assert(output({1, 0, 1}) == 23);
  assert(output({1, 1, 0}) == 39);
  assert(output({1, 1, 1}) == 47);
}

void test_bn() {
  tensor<3> input({2, 2, 2});
  for (int c = 0; c < input.dims_[0]; ++c) {
    for (int h = 0; h < input.dims_[1]; ++h) {
      for (int w = 0; w < input.dims_[2]; ++w) {
        input({c, h, w}) = c * input.dims_[1] * input.dims_[2] + h * input.dims_[2] + w;
      }
    }
  }
  tensor<1> mean({2});
  mean({0}) = 1;
  mean({1}) = 2;
  tensor<1> variance({2});
  variance({0}) = 4;
  variance({1}) = 16;
  tensor<1> scale({2});
  scale({0}) = 3;
  scale({1}) = 6;
  tensor<1> offset({2});
  offset({0}) = 0.5;
  offset({1}) = 1.5;
  tensor<3> output = batch_norm(input, mean, variance, scale, offset, 0);

  assert(output.dims_[0] == 2);
  assert(output.dims_[1] == 2);
  assert(output.dims_[2] == 2);

  assert(output({0, 0, 0}) == -1);
  assert(output({0, 0, 1}) == 0.5);
  assert(output({0, 1, 0}) == 2);
  assert(output({0, 1, 1}) == 3.5);

  assert(output({1, 0, 0}) == 4.5);
  assert(output({1, 0, 1}) == 6);
  assert(output({1, 1, 0}) == 7.5);
  assert(output({1, 1, 1}) == 9);
}

void test_conv_transpose1() {
  tensor<3> input({2, 2, 2});
  for (int c = 0; c < input.dims_[0]; ++c) {
    for (int h = 0; h < input.dims_[1]; ++h) {
      for (int w = 0; w < input.dims_[2]; ++w) {
        input({c, h, w}) = c * input.dims_[1] * input.dims_[2] + h * input.dims_[2] + w;
      }
    }
  }
  tensor<4> kernel({2, 1, 2, 2});
  for (int f = 0; f < kernel.dims_[0]; ++f) {
    for (int h = 0; h < kernel.dims_[2]; ++h) {
      for (int w = 0; w < kernel.dims_[3]; ++w) {
        kernel({f, 0, h, w}) = f * kernel.dims_[2] * kernel.dims_[3] + h * kernel.dims_[3] + w;
      }
    }
  }
  tensor<1> bias({1}, 1);
  tensor<3> output = conv_transpose(input, kernel, bias);

  assert(output.dims_[0] == 1);
  assert(output.dims_[1] == 4);
  assert(output.dims_[2] == 4);

  assert(output({0, 0, 0}) == 17);
  assert(output({0, 0, 1}) == 21);
  assert(output({0, 1, 0}) == 25);
  assert(output({0, 1, 1}) == 29);

  assert(output({0, 0, 2}) == 21);
  assert(output({0, 0, 3}) == 27);
  assert(output({0, 1, 2}) == 33);
  assert(output({0, 1, 3}) == 39);

  assert(output({0, 2, 0}) == 25);
  assert(output({0, 2, 1}) == 33);
  assert(output({0, 3, 0}) == 41);
  assert(output({0, 3, 1}) == 49);

  assert(output({0, 2, 2}) == 29);
  assert(output({0, 2, 3}) == 39);
  assert(output({0, 3, 2}) == 49);
  assert(output({0, 3, 3}) == 59);
}

void test_conv_transpose2() {
  tensor<3> input({1, 2, 2});
  for (int h = 0; h < input.dims_[1]; ++h) {
    for (int w = 0; w < input.dims_[2]; ++w) {
      input({0, h, w}) = h * input.dims_[2] + w;
    }
  }
  tensor<4> kernel({1, 2, 2, 2});
  for (int f = 0; f < kernel.dims_[1]; ++f) {
    for (int h = 0; h < kernel.dims_[2]; ++h) {
      for (int w = 0; w < kernel.dims_[3]; ++w) {
        kernel({0, f, h, w}) = f * kernel.dims_[2] * kernel.dims_[3] + h * kernel.dims_[3] + w;
      }
    }
  }
  tensor<1> bias({2});
  bias({0}) = 1;
  bias({1}) = 2;
  tensor<3> output = conv_transpose(input, kernel, bias);

  assert(output.dims_[0] == 2);
  assert(output.dims_[1] == 4);
  assert(output.dims_[2] == 4);

  for (int h = 0; h < 2; ++h) {
    for (int w = 0; w < 2; ++w) {
      if (h + w < 2) {
        assert(output({0, h, w}) == bias({0}));
        assert(output({1, h, w}) == bias({1}));
      }
    }
  }

  tensor<2> row_sums({2, 4});
  for (int f = 0; f < 2; ++f) {
    for (int h = 0; h < 4; ++h) {
      row_sums({f, h}) = 0;
      for (int w = 0; w < 4; ++w) {
        row_sums({f, h}) += output({f, h, w});
      }
    }
  }
  assert(row_sums({0, 0}) == 5);
  assert(row_sums({0, 1}) == 9);

  assert(row_sums({1, 2}) == 53);
  assert(row_sums({1, 3}) == 73);

  tensor<2> col_sums({2, 4});
  for (int f = 0; f < 2; ++f) {
    for (int w = 0; w < 4; ++w) {
      col_sums({f, w}) = 0;
      for (int h = 0; h < 4; ++h) {
        col_sums({f, w}) += output({f, h, w});
      }
    }
  }
  assert(col_sums({0, 2}) == 12);
  assert(col_sums({0, 3}) == 20);

  assert(col_sums({1, 0}) == 28);
  assert(col_sums({1, 1}) == 32);
}

void test_load_weights() {
  std::ifstream file("weights.bin", std::ios::binary);
  tensor<4> weights({64, 3, 3, 3});
  load_weights(file, weights);
  assert(weights.dims_[0] == 64);
  assert(weights.dims_[1] == 3);
  assert(weights.dims_[2] == 3);
  assert(weights.dims_[3] == 3);
  assert(std::fabs(weights({0, 0, 0, 0}) - 0.1244) < epsilon);
  assert(std::fabs(weights.data_[weights.size_ - 1] - 0.0135) < epsilon);
}

void test_read_ppm() {
  tensor<3> image = read_ppm("car.ppm");
  assert(image.dims_[0] == 3);
  assert(image.dims_[1] == 256);
  assert(image.dims_[2] == 512);

  assert(image({0, 0, 0}) == 238);

  tensor<2> row_sums({3, 256});
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < 256; ++h) {
      row_sums({c, h}) = 0;
      for (int w = 0; w < 512; ++w) {
        row_sums({c, h}) += image({c, h, w});
      }
    }
  }
  assert(std::fabs(row_sums({0, 0}) - 114156) < 1e5);
  assert(std::fabs(row_sums({1, 0}) - 112164) < 1e5);
  assert(std::fabs(row_sums({2, 0}) - 111305) < 1e5);

  assert(std::fabs(row_sums({0, 255}) - 85117) < 1e5);
  assert(std::fabs(row_sums({1, 255}) - 82237) < 1e5);
  assert(std::fabs(row_sums({2, 255}) - 77525) < 1e5);

  tensor<2> col_sums({3, 512});
  for (int c = 0; c < 3; ++c) {
    for (int w = 0; w < 512; ++w) {
      col_sums({c, w}) = 0;
      for (int h = 0; h < 256; ++h) {
        col_sums({c, w}) += image({c, h, w});
      }
    }
  }
  assert(std::fabs(col_sums({0, 0}) - 47306) < 1e5);
  assert(std::fabs(col_sums({1, 0}) - 46836) < 1e5);
  assert(std::fabs(col_sums({2, 0}) - 47070) < 1e5);

  assert(std::fabs(col_sums({0, 511}) - 49114) < 1e5);
  assert(std::fabs(col_sums({1, 511}) - 48087) < 1e5);
  assert(std::fabs(col_sums({2, 511}) - 47715) < 1e5);
}

void test_unet1() {
  unet model(3, 21);
  tensor<3> input({3, 256, 256});
  tensor<3> output = model(input);

  assert(output.dims_[0] == 21);
  assert(output.dims_[1] == 256);
  assert(output.dims_[2] == 256);
}

void test_unet2() {
  unet model(3, 2);
  model.load_checkpoint("weights.bin");

  tensor<3> input({3, 16, 32}, 0);
  tensor<3> output = model(input);

  // std::cout << output({0, 0, 0}) << std::endl;
  assert(std::fabs(output({0, 0, 0}) - 0.4432) < epsilon);

  tensor<1> channel_sums({2});
  for (int c = 0; c < 2; ++c) {
    channel_sums({c}) = 0;
    for (int h = 0; h < 16; ++h) {
      for (int w = 0; w < 32; ++w) {
        channel_sums({c}) += output({c, h, w});
      }
    }
  }
  assert(std::fabs(channel_sums({0}) - 529.7327) < epsilon);
  assert(std::fabs(channel_sums({1}) + 771.4091) < epsilon);

  tensor<1> row_sums_0({16});
  for (int h = 0; h < 16; ++h) {
    row_sums_0({h}) = 0;
    for (int w = 0; w < 32; ++w) {
      row_sums_0({h}) += output({0, h, w});
    }
  }
  assert(std::fabs(row_sums_0({0}) - 10.5315) < epsilon);
  assert(std::fabs(row_sums_0({15}) - 42.6228) < epsilon);

  tensor<1> col_sums_1({32});
  for (int w = 0; w < 32; ++w) {
    col_sums_1({w}) = 0;
    for (int h = 0; h < 16; ++h) {
      col_sums_1({w}) += output({1, h, w});
    }
  }
  assert(std::fabs(col_sums_1({0}) + 18.1114) < epsilon);
  assert(std::fabs(col_sums_1({31}) + 18.8850) < epsilon);
}

int main() {
  test_conv1();
  test_conv2();
  test_conv3();
  test_bn();
  test_conv_transpose1();
  test_conv_transpose2();
  test_load_weights();
  test_unet1();
  test_unet2();
  test_read_ppm();

  return 0;
}
