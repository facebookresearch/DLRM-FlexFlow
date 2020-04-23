#pragma warning disable
#include "model.h"
#include "cuda_helper.h"

template <int DIM>
Tensor FFModel::sigmoid(std::string name, const Tensor& input, const int output_shape[])
{
  Activation<DIM> *activation = new Activation<DIM>(*this, name, "sigmoid", input, output_shape);
  layers.push_back(activation);
  return activation->output;
}
template <int DIM>
Tensor FFModel::relu(std::string name, const Tensor& input, const int output_shape[])
{
  Activation<DIM> *activation = new Activation<DIM>(*this, name, "relu", input, output_shape);
  layers.push_back(activation);
  return activation->output;
}
template <int DIM>
Tensor FFModel::elu(std::string name, const Tensor& input, const int output_shape[])
{
  Activation<DIM> *activation = new Activation<DIM>(*this, name, "elu", input, output_shape);
  layers.push_back(activation);
  return activation->output;
}
template <int DIM>
Tensor FFModel::identity(std::string name, const Tensor& input, const int output_shape[])
{
  Activation<DIM> *activation = new Activation<DIM>(*this, name, "identity", input, output_shape);
  layers.push_back(activation);
  return activation->output;
}
template <int DIM>
Tensor FFModel::tanh(std::string name, const Tensor& input, const int output_shape[])
{
  Activation<DIM> *activation = new Activation<DIM>(*this, name, "tanh", input, output_shape);
  layers.push_back(activation);
  return activation->output;
}



template <int DIM>
Activation<DIM>::Activation(FFModel& model,
  const std::string& pcname,
  const std::string& _mode,
  const Tensor& _input, 
  const int output_shape[])
  : Op(pcname, _input)
{
  task_is = IndexSpaceT<DIM>(model.get_or_create_task_is(DIM, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<DIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Create output tensor
  output = model.create_tensor<DIM>(output_shape, task_is, DT_FLOAT);
  model.create_data_parallel_partition_with_diff_dims<DIM, DIM>(
      _input, task_is, input_lps[0], input_grad_lps[0]);
  mode = _mode;
}

template <int DIM>
void Activation<DIM>::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    if (mode == "relu") {
      handle.mode = CUDNN_ACTIVATION_RELU;
    } else if (mode == "sigmoid") {
      handle.mode = CUDNN_ACTIVATION_SIGMOID;
    } else if (mode == "elu") {
      handle.mode = CUDNN_ACTIVATION_ELU;
    } else if (mode == "tanh") {
      handle.mode = CUDNN_ACTIVATION_TANH;
    } else if (mode == "idendity") {
      handle.mode = CUDNN_ACTIVATION_IDENTITY;
    } else {
      throw 255;
    }
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  auto task_id = ACTIVATION_3D_INIT_TASK_ID;
  if (DIM == 3) {
    task_id = ACTIVATION_3D_INIT_TASK_ID;
  } else if (DIM == 2) {
    task_id = ACTIVATION_2D_INIT_TASK_ID;
  } else if (DIM == 1) {
    task_id = ACTIVATION_1D_INIT_TASK_ID;
  }
  else {
    printf("idim %d odim %d not supported", DIM, DIM);
  }
  IndexLauncher launcher(task_id, task_is,
    TaskArgument(this, sizeof(Activation)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_WRITE, EXCLUSIVE, inputs[0].region));



  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_DISCARD, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

template <int DIM>
OpMeta* Activation<DIM>::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handle = *((const FFHandler*) task->local_args);
  ActivationMeta* m = new ActivationMeta(handle);
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, DIM> acc_input(regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, DIM> acc_output(regions[1], task->regions[1], FID_DATA, ctx, runtime);

#ifndef DISABLE_COMPUTATION
  // assert(rect_input == rect_output);
  int dims[DIM];
  int dims_buf[DIM];
  int stride[DIM];
  int stride_buf[DIM];
  stride_buf[0] = 1;
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  checkCUDNN(cudnnCreateActivationDescriptor(&m->activation));
  checkCUDNN(cudnnSetActivationDescriptor(
    m->activation,
    // CUDNN_ACTIVATION_TANH,
    m->handle.mode,
    CUDNN_NOT_PROPAGATE_NAN,
    0.0
  ));   
  if (DIM == 1) {
    int batch_size = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
    checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size, 1, 1, 1));
  }
  else if (DIM == 2) {
    int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
    int batch_size = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
    checkCUDNN(cudnnSetTensor4dDescriptor(m->inputTensor,
      CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT,
      batch_size, in_dim, 1, 1));
      // 1, 1, batch_size, in_dim));
  }
  else if (DIM > 2) {
    // cuda tensor dims order from outer to inner , so dims[0] is batch_dimension
    for (int i = 0; i < DIM; i++) {
      dims_buf[i] = acc_input.rect.hi[i] - acc_input.rect.lo[i] + 1;
      if (i + 1 < DIM) {
        stride_buf[i+1] = stride_buf[i] * dims_buf[i];
      }
    }
    for (int i = 0; i < DIM; i++) {
      dims[i] = dims_buf[DIM-i-1];
      stride[i] = stride_buf[DIM-i-1];
    }
    /*
    https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html#cudnnSetTensorNdDescriptor
    Note: Do not use for 2 dimensional tensors. The minimum number of dimensions in the filter descriptor is three. For more information, see cudnnGetRNNLinLayerBiasParams().
    */
    checkCUDNN(cudnnSetTensorNdDescriptor(m->inputTensor,
                                          CUDNN_DATA_FLOAT,
                                          DIM,
                                          dims,
                                          stride));
                                                 
  }
                              
#endif
  return m;
}





/*
  regions[0](I): input
  regions[1](O): output
*/  
template <int DIM>
void Activation<DIM>::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  const ActivationMeta* m = *((ActivationMeta**) task->local_args);
  TensorAccessorR<float, DIM> acc_input(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, DIM> acc_output(
    regions[1], task->regions[1], FID_DATA, ctx, runtime,
    false/*readOutput*/);
#ifndef DISABLE_LEGION_CUDA_HIJACK
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif
  // DOUBLE CHECK HANDLE TO PREVENT SEGMENTATION FAULT
  checkCUDA(cudnnActivationForward(
    m->handle.dnn,
    m->activation,
    &alpha, m->inputTensor, acc_input.ptr,
    &beta, m->inputTensor, acc_output.ptr
  ));

}

template <int DIM>
void Activation<DIM>::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  auto task_id = ACTIVATION_3D_FWD_TASK_ID;
  if (DIM == 3) {
    task_id = ACTIVATION_3D_FWD_TASK_ID;
  } else if (DIM == 2) {
    task_id = ACTIVATION_2D_FWD_TASK_ID;
  } else if (DIM == 1) {
    task_id = ACTIVATION_1D_FWD_TASK_ID;
  } else {
    printf("idim %d odim %d not supported", DIM, DIM);
  }
  IndexLauncher launcher(task_id, task_is,
    TaskArgument(NULL, 0), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I) : input
  regions[1](I) : output
  regions[2](O) : input_grad
  regions[3](I) : output_grad
*/
template <int DIM>
void Activation<DIM>::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  const ActivationMeta* m = *((ActivationMeta**) task->local_args);
  TensorAccessorR<float, DIM> acc_input(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, DIM> acc_output(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, DIM> acc_output_grad(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);
  TensorAccessorW<float, DIM> acc_input_grad(
    regions[2], task->regions[2], FID_DATA, ctx, runtime,
    false/*readOutput*/);


#ifndef DISABLE_LEGION_CUDA_HIJACK
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDNN(cudnnSetStream(m->handle.dnn, stream));
#endif

  checkCUDA(cudnnActivationBackward(
    m->handle.dnn,
    m->activation,
    &alpha, 
    m->inputTensor, acc_output.ptr,
    m->inputTensor, acc_output_grad.ptr,
    m->inputTensor, acc_input.ptr,
    &beta, m->inputTensor, acc_input_grad.ptr
  ));
}

template <int DIM>
void Activation<DIM>::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  auto task_id = ACTIVATION_3D_BWD_TASK_ID;
  if (DIM == 3) {
    task_id = ACTIVATION_3D_BWD_TASK_ID;
  } else if (DIM == 2) {
    task_id = ACTIVATION_2D_BWD_TASK_ID;
  } else if (DIM == 1) {
    task_id = ACTIVATION_1D_BWD_TASK_ID;
  } else {
    printf("idim %d odim %d not supported", DIM, DIM);
  }
  IndexLauncher launcher(task_id, task_is,
    TaskArgument(NULL, 0), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[0], 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


template Activation<1>::Activation(FFModel& model,
  const std::string& pcname,
  const std::string& _mode,
  const Tensor& _input,
  const int output_shape[]);
template Activation<2>::Activation(FFModel& model,
  const std::string& pcname,
  const std::string& _mode,
  const Tensor& _input,
  const int output_shape[]);
template Activation<3>::Activation(FFModel& model,
  const std::string& pcname,
  const std::string& _mode,
  const Tensor& _input,
  const int output_shape[]);
template OpMeta* Activation<1>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);  
template OpMeta* Activation<2>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template OpMeta* Activation<3>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<1>::init(const FFModel& ff);
template void Activation<2>::init(const FFModel& ff);
template void Activation<3>::init(const FFModel& ff);
template void Activation<1>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<2>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<3>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<1>::forward(const FFModel& ff);
template void Activation<2>::forward(const FFModel& ff);
template void Activation<3>::forward(const FFModel& ff);
template void Activation<1>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<2>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<3>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Activation<1>::backward(const FFModel& ff);
template void Activation<2>::backward(const FFModel& ff);
template void Activation<3>::backward(const FFModel& ff);

template Tensor FFModel::relu<3>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::relu<2>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::relu<1>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::elu<3>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::elu<2>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::elu<1>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::sigmoid<3>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::sigmoid<2>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::sigmoid<1>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::identity<3>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::identity<2>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::identity<1>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::tanh<3>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::tanh<2>(std::string name, const Tensor& input, const int output_shape[]);
template Tensor FFModel::tanh<1>(std::string name, const Tensor& input, const int output_shape[]);
