#pragma once

/*
Modified from tiny-cuda-nn/bindings/torch/tinycudann/bindings.cpp
Ref:
- https://pytorch.org/tutorials/advanced/cpp_autograd.html
- https://pytorch.org/cppdocs/api/structtorch_1_1autograd_1_1_function.html
- https://github.com/Totoro97/f2-nerf/blob/main/src/Field/TCNNWP.h
-
https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html#implementing-and-binding-the-class-in-c
 */
#include "bindings.h"
#include <tiny-cuda-nn/cpp_api.h>
#include <torch/torch.h>

struct TCNNModule : torch::nn::Module {
  TCNNModule() = default;
  TCNNModule(std::shared_ptr<tcnn_binding::Module> _p_native_tcnn_module,
                const size_t &_n_input_dims,
                const std::string &_name = "params", const int &_seed = 1337);
  void init_module(std::shared_ptr<tcnn_binding::Module> _p_native_tcnn_module,
                   const size_t &_n_input_dims,
                   const std::string &_name = "params",
                   const int &_seed = 1337);

  virtual size_t get_out_dim() const;

  std::shared_ptr<tcnn_binding::Module> p_native_tcnn_module_;
  torch::ScalarType dtype_;
  int seed_;
  torch::Tensor params_;
  std::string name_;
  float loss_scale_;

  size_t n_input_dims_;
  size_t n_output_dims_;

  tcnn::cpp::Context tcnn_ctx_;

  torch::Tensor forward(const torch::Tensor &x);
};

class TCNNInfo : public torch::CustomClassHolder {
public:
  tcnn_binding::Module *native_tcnn_module_;

  tcnn::cpp::Context native_ctx_;
  torch::autograd::AutogradContext *ctx_fwd_;
  float loss_scale_;
};

class TCNNModuleFunction
    : public torch::autograd::Function<TCNNModuleFunction> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor _input, torch::Tensor _params,
                               torch::IValue _binding_info);

  // TODO: there is possible double backward, check _module_function_backward
  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs);
};

class TCNNModuleFunctionBackward
    : public torch::autograd::Function<TCNNModuleFunctionBackward> {
public:
  static torch::autograd::tensor_list
  forward(torch::autograd::AutogradContext *ctx, torch::Tensor _doutput,
          torch::Tensor _input, torch::Tensor _params, torch::Tensor _output,
          torch::IValue _binding_info);

  static torch::autograd::tensor_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs);
};

struct TCNNNetwork : TCNNModule {
  TCNNNetwork() = default;
  TCNNNetwork(size_t _n_input_dims, size_t _n_output_dims,
                 const tcnn::cpp::json &_network_config,
                 const std::string &_name = "network_params",
                 const int &_seed = 1337);

  void init_network(size_t _n_input_dims, size_t _n_output_dims,
                    const tcnn::cpp::json &_network_config,
                    const std::string &_name = "network_params",
                    const int &_seed = 1337);

  tcnn::cpp::json network_config_;
};

struct TCNNEncoding : TCNNModule {
  TCNNEncoding() = default;
  TCNNEncoding(size_t _n_input_dims, const tcnn::cpp::json &_encoding_config,
                  const std::string &_name = "encoding_params",
                  const int &_seed = 1337);

  void init_encoding(size_t _n_input_dims,
                     const tcnn::cpp::json &_encoding_config,
                     const std::string &_name = "encoding_params",
                     const int &_seed = 1337);

  tcnn::cpp::json encoding_config_;
};