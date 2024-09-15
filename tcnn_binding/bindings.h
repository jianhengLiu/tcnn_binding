/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** @file   torch_bindings.cu
 *  @author Thomas Müller, Jacob Munkberg, Jon Hasselgren, Or Perel, NVIDIA
 */
#pragma once

#include <torch/torch.h>

#include <ATen/cuda/CUDAUtils.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef snprintf
#undef snprintf
#endif

#include <json/json.hpp>

#include <tiny-cuda-nn/cpp_api.h>

#define STRINGIFY(x) #x
#define STR(x) STRINGIFY(x)
#define FILE_LINE __FILE__ ":" STR(__LINE__)
#define CHECK_THROW(x)                                                         \
  do {                                                                         \
    if (!(x))                                                                  \
      throw std::runtime_error(std::string(FILE_LINE " check failed " #x));    \
  } while (0)

namespace tcnn_binding {

c10::ScalarType torch_type(tcnn::cpp::Precision precision);

void *void_data_ptr(torch::Tensor &tensor);

class Module {
public:
  Module(tcnn::cpp::Module *module);

  std::tuple<tcnn::cpp::Context, torch::Tensor> fwd(torch::Tensor input,
                                                    torch::Tensor params);

  std::tuple<torch::Tensor, torch::Tensor>
  bwd(const tcnn::cpp::Context &ctx, torch::Tensor input, torch::Tensor params,
      torch::Tensor output, torch::Tensor dL_doutput);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  bwd_bwd_input(const tcnn::cpp::Context &ctx, torch::Tensor input,
                torch::Tensor params, torch::Tensor dL_ddLdinput,
                torch::Tensor dL_doutput);

  torch::Tensor initial_params(size_t seed);

  uint32_t n_input_dims() const;

  uint32_t n_params() const;
  tcnn::cpp::Precision param_precision() const;
  c10::ScalarType c10_param_precision() const;

  uint32_t n_output_dims() const;
  tcnn::cpp::Precision output_precision() const;
  c10::ScalarType c10_output_precision() const;

  nlohmann::json hyperparams() const;
  std::string name() const;

private:
  std::unique_ptr<tcnn::cpp::Module> m_module;
};

#if !defined(TCNN_NO_NETWORKS)
Module create_network_with_input_encoding(uint32_t n_input_dims,
                                          uint32_t n_output_dims,
                                          const nlohmann::json &encoding,
                                          const nlohmann::json &network);

Module create_network(uint32_t n_input_dims, uint32_t n_output_dims,
                      const nlohmann::json &network);
#endif

Module create_encoding(uint32_t n_input_dims, const nlohmann::json &encoding,
                       tcnn::cpp::Precision requested_precision);

} // namespace tcnn_binding