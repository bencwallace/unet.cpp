#include <cassert>
#include <limits>

#include "ops.h"

tensor<2> im2col(const tensor<3> &im, int kernel_height, int kernel_width, int padding) {
  int col_height = im.dims_[1] + 2 * padding - kernel_height + 1;
  int col_width = im.dims_[2] + 2 * padding - kernel_width + 1;
  tensor<2> col({kernel_width * kernel_height * im.dims_[0], col_height * col_width});
#pragma omp parallel for
  for (int d = 0; d < im.dims_[0]; ++d) {
    for (int i = 0; i < kernel_height; ++i) {
      for (int j = 0; j < kernel_width; ++j) {
        for (int h = 0; h < col_height; ++h) {
          for (int w = 0; w < col_width; ++w) {
            int h_in = h - padding + i;
            int w_in = w - padding + j;
            auto &val = col({d * kernel_height * kernel_width + i * kernel_width + j, h * col_width + w});
            if (h_in >= 0 && h_in < im.dims_[1] && w_in >= 0 && w_in < im.dims_[2]) {
              val = im({d, h_in, w_in});
            } else {
              val = 0;
            }
          }
        }
      }
    }
  }
  return col;
}

tensor<3> col2im(const tensor<2> &col, int num_filters, int kernel_height, int kernel_width, int im_height,
                 int im_width) {
  tensor<3> im({num_filters, im_height, im_width});
  int num_window_rows = im_height / kernel_height;
  int num_window_cols = im_width / kernel_width;

  for (int d = 0; d < im.dims_[0]; ++d) {
    for (int i = 0; i < kernel_height; ++i) {
      for (int j = 0; j < kernel_width; ++j) {
        for (int h = 0; h < num_window_rows; ++h) {
          for (int w = 0; w < num_window_cols; ++w) {
            int h_out = h * kernel_height + i;
            int w_out = w * kernel_width + j;
            im({d, h_out, w_out}) =
                col({d * kernel_height * kernel_width + i * kernel_width + j, h * num_window_cols + w});
          }
        }
      }
    }
  }
  return im;
}

tensor<3> conv(const tensor<3> &input, const tensor<4> &kernel, const tensor<1> &bias, int padding) {
  int output_height = input.dims_[1] + 2 * padding - kernel.dims_[2] + 1;
  int output_width = input.dims_[2] + 2 * padding - kernel.dims_[3] + 1;
  tensor<3> output({kernel.dims_[0], output_height, output_width});

  tensor<2> col = im2col(input, kernel.dims_[2], kernel.dims_[3], padding);

#pragma omp parallel for
  for (int f = 0; f < kernel.dims_[0]; ++f) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        output({f, h, w}) = bias({f});
      }
    }
    for (int d = 0; d < kernel.dims_[1]; ++d) {
      for (int i = 0; i < kernel.dims_[2]; ++i) {
        for (int j = 0; j < kernel.dims_[3]; ++j) {
          float k = kernel({f, d, i, j});
          for (int h = 0; h < output_height; ++h) {
            for (int w = 0; w < output_width; ++w) {
              output({f, h, w}) +=
                  col({d * kernel.dims_[2] * kernel.dims_[3] + i * kernel.dims_[3] + j, h * output_width + w}) * k;
            }
          }
        }
      }
    }
  }
  return output;
}

tensor<3> conv_transpose(const tensor<3> &input, const tensor<4> &kernel, const tensor<1> &bias) {
  // require kernel.dims_[0] == input.dims_[0]
  // then output.dims_[0] == kernel.dims_[1]
  int out_height = input.dims_[1] * kernel.dims_[2];
  int out_width = input.dims_[2] * kernel.dims_[3];

  int out_window_vol = kernel.dims_[1] * kernel.dims_[2] * kernel.dims_[3];
  int out_num_windows = input.dims_[1] * input.dims_[2];
  tensor<2> col({out_window_vol, out_num_windows}, 0);

#pragma omp parallel for
  for (int f = 0; f < kernel.dims_[1]; ++f) {
    for (int d = 0; d < kernel.dims_[0]; ++d) {
      for (int i = 0; i < kernel.dims_[2]; ++i) {
        for (int j = 0; j < kernel.dims_[3]; ++j) {
          float k = kernel({d, f, i, j});
          for (int h = 0; h < input.dims_[1]; ++h) {
            for (int w = 0; w < input.dims_[2]; ++w) {
              col({f * kernel.dims_[2] * kernel.dims_[3] + i * kernel.dims_[3] + j, h * input.dims_[2] + w}) +=
                  input({d, h, w}) * k;
            }
          }
        }
      }
    }
  }

  tensor<3> output = col2im(col, kernel.dims_[1], kernel.dims_[2], kernel.dims_[3], out_height, out_width);
  for (int f = 0; f < kernel.dims_[1]; ++f) {
    for (int h = 0; h < out_height; ++h) {
      for (int w = 0; w < out_width; ++w) {
        output({f, h, w}) += bias({f});
      }
    }
  }
  return output;
}

tensor<3> max_pool(const tensor<3> &input, int size) {
  int output_height = input.dims_[1] / size;
  int output_width = input.dims_[2] / size;
  tensor<3> output({input.dims_[0], output_height, output_width});
  for (int f = 0; f < input.dims_[0]; ++f) {
    for (int h = 0; h < output_height; ++h) {
      for (int w = 0; w < output_width; ++w) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < size; ++i) {
          for (int j = 0; j < size; ++j) {
            max_val = std::max(max_val, input({f, h * size + i, w * size + j}));
          }
        }
        output({f, h, w}) = max_val;
      }
    }
  }
  return output;
}

tensor<3> cat(const tensor<3> &a, const tensor<3> &b) {
  assert(a.dims_[1] == b.dims_[1]);
  assert(a.dims_[2] == b.dims_[2]);
  tensor<3> output({a.dims_[0] + b.dims_[0], a.dims_[1], a.dims_[2]});
  for (int f = 0; f < a.dims_[0]; ++f) {
    for (int h = 0; h < a.dims_[1]; ++h) {
      for (int w = 0; w < a.dims_[2]; ++w) {
        output({f, h, w}) = a({f, h, w});
      }
    }
  }
  for (int f = 0; f < b.dims_[0]; ++f) {
    for (int h = 0; h < b.dims_[1]; ++h) {
      for (int w = 0; w < b.dims_[2]; ++w) {
        output({a.dims_[0] + f, h, w}) = b({f, h, w});
      }
    }
  }
  return output;
}

tensor<2> argmax(const tensor<3> &input) {
  tensor<2> output({input.dims_[1], input.dims_[2]});
  for (int h = 0; h < input.dims_[1]; ++h) {
    for (int w = 0; w < input.dims_[2]; ++w) {
      float max_val = -std::numeric_limits<float>::infinity();
      int max_idx = -1;
      for (int f = 0; f < input.dims_[0]; ++f) {
        if (input({f, h, w}) > max_val) {
          max_val = input({f, h, w});
          max_idx = f;
        }
      }
      output({h, w}) = max_idx;
    }
  }
  return output;
}
