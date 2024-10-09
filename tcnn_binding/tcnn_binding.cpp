#include "tcnn_binding.h"

/* BindingModule */
TCNNModule::TCNNModule(
    std::shared_ptr<tcnn_binding::Module> _p_native_tcnn_module,
    const size_t &_n_input_dims, const std::string &_name, const int &_seed)
    : torch::nn::Module() {
  init_module(p_native_tcnn_module_, n_input_dims_, name_, seed_);
}

void TCNNModule::init_module(
    std::shared_ptr<tcnn_binding::Module> _p_native_tcnn_module,
    const size_t &_n_input_dims, const std::string &_name, const int &_seed) {
  p_native_tcnn_module_ = std::move(_p_native_tcnn_module);
  n_input_dims_ = _n_input_dims;
  name_ = _name;
  seed_ = _seed;

  dtype_ = tcnn_binding::torch_type(p_native_tcnn_module_->param_precision());
  params_ = p_native_tcnn_module_->initial_params(seed_);
  loss_scale_ = default_loss_scale(p_native_tcnn_module_->param_precision());
}

size_t TCNNModule::get_out_dim() const { return n_output_dims_; }

torch::Tensor TCNNModule::forward(const torch::Tensor &x) {
  if (!x.is_cuda()) {
    std::cout << "BindingModule::forward: input is not on CUDA\n";
  }

  auto batch_size = x.size(0);
  auto batch_size_granularity = int(tcnn::cpp::batch_size_granularity());
  long padded_batch_size = (batch_size + batch_size_granularity - 1) /
                           batch_size_granularity * batch_size_granularity;

  auto x_padded =
      batch_size == padded_batch_size
          ? x
          : torch::nn::functional::pad(
                x, torch::nn::functional::PadFuncOptions(
                       {0LL, 0LL, 0LL,
                        (long long)(padded_batch_size - batch_size)}));

  auto p_binding_info = torch::make_intrusive<TCNNInfo>();
  p_binding_info->native_tcnn_module_ = p_native_tcnn_module_.get();
  p_binding_info->loss_scale_ = loss_scale_;
  auto output = TCNNModuleFunction::apply(
      x_padded.to(torch::kFloat).contiguous(),
      params_
          .to(tcnn_binding::torch_type(
              p_native_tcnn_module_->param_precision()))
          .contiguous(),
      torch::IValue(p_binding_info));
  return output
      .index({torch::indexing::Slice(0, batch_size),
              torch::indexing::Slice(0, n_output_dims_)})
      .to(torch::kFloat32);
}

/* BindingModuleFunction */

TORCH_LIBRARY(tcnn_binding, m) {
  m.class_<TCNNInfo>("TCNNInfo").def(torch::init());
}

torch::Tensor null_tensor_like(const torch::Tensor &tensor) {
  return torch::empty(
      {}, torch::TensorOptions().dtype(tensor.dtype()).device(tensor.device()));
}

torch::Tensor null_tensor_to_none(torch::Tensor &tensor) {
  if (tensor.sizes().empty()) {
    return torch::Tensor();
  }
  return tensor;
}

torch::Tensor TCNNModuleFunction::forward(torch::autograd::AutogradContext *ctx,
                                          const torch::Tensor &_input,
                                          const torch::Tensor &_params,
                                          const torch::IValue &_binding_info) {
  ctx->set_materialize_grads(false);

  auto p_binding_info = _binding_info.toCustomClass<TCNNInfo>();
  auto [native_ctx, output] =
      p_binding_info->native_tcnn_module_->fwd(_input, _params);

  ctx->save_for_backward({_input, _params, output});
  p_binding_info->native_ctx_ = std::move(native_ctx);
  ctx->saved_data["binding_info"] = torch::IValue(p_binding_info);
  return {output};
}

torch::autograd::tensor_list
TCNNModuleFunction::backward(torch::autograd::AutogradContext *ctx,
                             torch::autograd::variable_list grad_outputs) {
  auto doutput = grad_outputs[0];
  // check if doutput is empty tensor
  if (!doutput.defined()) {
    return {torch::Tensor(), torch::Tensor(), torch::Tensor()};
  }

  if (!doutput.is_cuda()) {
    std::cout << "BindingModuleFunction::backward: doutput is not on CUDA\n";
    doutput = doutput.cuda();
  }
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  auto params = saved[1];
  auto output = saved[2];

  auto tuple_output = TCNNModuleFunctionBackward::apply(
      doutput, input, params, output, ctx->saved_data["binding_info"]);

  auto input_grad = tuple_output[0];
  auto params_grad = tuple_output[1];
  return {null_tensor_to_none(input_grad), null_tensor_to_none(params_grad),
          torch::Tensor()};
}

/* BindingModuleFunctionBackward */
torch::autograd::tensor_list TCNNModuleFunctionBackward::forward(
    torch::autograd::AutogradContext *ctx, const torch::Tensor &_doutput,
    const torch::Tensor &_input, const torch::Tensor &_params,
    const torch::Tensor &_output, const torch::IValue &_binding_info) {

  ctx->save_for_backward({_input, _params, _doutput});
  ctx->saved_data["binding_info"] = _binding_info;
  // with torch.no_grad():
  auto p_binding_info = _binding_info.toCustomClass<TCNNInfo>();

  torch::NoGradGuard no_grad;
  auto scaled_grad = _doutput * p_binding_info->loss_scale_;

  auto [input_grad, params_grad] = p_binding_info->native_tcnn_module_->bwd(
      p_binding_info->native_ctx_, _input, _params, _output, scaled_grad);

  if (input_grad.defined()) {
    input_grad = input_grad / p_binding_info->loss_scale_;
  } else {
    input_grad = null_tensor_like(_input);
  }
  if (params_grad.defined()) {
    params_grad = params_grad / p_binding_info->loss_scale_;
  } else {
    params_grad = null_tensor_like(_params);
  }
  return {input_grad, params_grad};
}

torch::autograd::tensor_list TCNNModuleFunctionBackward::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  // # NOTE: currently support:
  // #       ✓   d(dL_dinput)_d(dL_doutput)  doutput_grad
  // #       ✓   d(dL_dinput)_d(params)      params_grad
  // #       ✓   d(dL_dinput)_d(input)       input_grad
  // #       x   d(dL_dparam)_d(...)
  auto saved_tensors = ctx->get_saved_variables();
  auto input = saved_tensors[0];
  auto params = saved_tensors[1];
  auto grad_output = saved_tensors[2];

  auto p_binding_info =
      ctx->saved_data["binding_info"].toCustomClass<TCNNInfo>();

  auto grad_mode = torch::GradMode::is_enabled();
  torch::GradMode::set_enabled(true);
  // # NOTE: preserves requires_grad info (this function is in no_grad() context
  // by default when invoking loss.backward())
  grad_output = grad_output * p_binding_info->loss_scale_;

  torch::GradMode::set_enabled(false);
  auto dinput_grad = grad_outputs[0];
  auto [doutput_grad, params_grad, input_grad] =
      p_binding_info->native_tcnn_module_->bwd_bwd_input(
          p_binding_info->native_ctx_, input, params, dinput_grad, grad_output);

  // # NOTE: be cautious when multiplying and dividing loss_scale
  // #       doutput_grad uses dinput_grad
  // #       params_grad  uses dinput_grad * doutput
  // #       input_grad   uses dinput_grad * doutput
  params_grad = params_grad.defined()
                    ? params_grad / p_binding_info->loss_scale_
                    : null_tensor_like(input);
  input_grad = input_grad.defined() ? input_grad / p_binding_info->loss_scale_
                                    : null_tensor_like(input);
  torch::GradMode::set_enabled(grad_mode);
  return {doutput_grad, input_grad, params_grad, torch::Tensor(),
          torch::Tensor()};
}

/* BindingNetwork */
TCNNNetwork::TCNNNetwork(size_t _n_input_dims, size_t _n_output_dims,
                         const tcnn::cpp::json &_network_config,
                         const std::string &_name, const int &_seed) {
  init_network(_n_input_dims, _n_output_dims, _network_config, _name, _seed);
}

void TCNNNetwork::init_network(size_t _n_input_dims, size_t _n_output_dims,
                               const tcnn::cpp::json &_network_config,
                               const std::string &_name, const int &_seed) {
  network_config_ = _network_config;
  init_module(
      std::make_shared<tcnn_binding::Module>(tcnn_binding::create_network(
          _n_input_dims, _n_output_dims, network_config_)),
      _n_input_dims, _name, _seed);

  // be carefull that p_native_tcnn_module_->n_output_dims() return padded
  // output dims
  n_output_dims_ = _n_output_dims;
}

/* BindingNetwork */
TCNNEncoding::TCNNEncoding(size_t _n_input_dims,
                           const tcnn::cpp::json &_encoding_config,
                           const std::string &_name, const int &_seed) {
  init_encoding(_n_input_dims, _encoding_config, _name, _seed);
}

void TCNNEncoding::init_encoding(size_t _n_input_dims,
                                 const tcnn::cpp::json &_encoding_config,
                                 const std::string &_name, const int &_seed) {
  encoding_config_ = _encoding_config;
  init_module(
      std::make_shared<tcnn_binding::Module>(tcnn_binding::create_encoding(
          _n_input_dims, encoding_config_, tcnn::cpp::preferred_precision())),
      _n_input_dims, _name, _seed);
  n_output_dims_ = p_native_tcnn_module_->n_output_dims();
}
