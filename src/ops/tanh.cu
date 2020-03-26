#pragma warning disable
#include "model.h"
#include "cuda_helper.h"

template <int DIM>
Tensor FFModel::tanh(std::string name, const Tensor& input)
{
  Tanh<DIM> *tanh = new Tanh<DIM>(*this, name, input);
  layers.push_back(tanh);
  return tanh->output;
}


template <int DIM>
Tanh<DIM>::Tanh(FFModel& model,
  const std::string& pcname,
  const Tensor& _input)
  : Op(pcname, _input)
{
  task_is = IndexSpaceT<DIM>(model.get_or_create_task_is(DIM, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<DIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  // Create output tensor
  output = model.create_tensor<DIM>(_input.adim, task_is, DT_FLOAT);
  model.create_data_parallel_partition_with_diff_dims<DIM, DIM>(
      _input, task_is, input_lps[0], input_grad_lps[0]);

}



template <int DIM>
OpMeta* Tanh<DIM>::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  TanhMeta* m = new TanhMeta(handler);
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Softmax* softmax = (Softmax*) task->args;
  const AccessorRO<float, DIM> acc_input(regions[0], FID_DATA);
  const AccessorWO<float, DIM> acc_output(regions[1], FID_DATA);
  Rect<DIM> rect_input, rect_output;
  rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_output.accessor.is_dense_arbitrary(rect_output));
#ifndef DISABLE_COMPUTATION
  checkCUDNN(cudnnCreateTensorDescriptor(&m->inputTensor));
  //checkCUDNN(cudnnCreateTensorDescriptor(&m->outputTensor));
  assert(rect_input == rect_output);
  int dims[DIM];
  int stride[DIM];
  stride[0] = 1;
  // hypothesis 1 cuda descriptor assumes the out most dimension is batch dimension
  // hypothesis 2 cuda descriptor assumes the inner most dimension is nbatch dimension
  for (int i = 0; i < DIM; i++) {
    dims[i] = rect_input.hi[i] - rect_input.lo[i] + 1;
    if (i + 1 < DIM) {
      stride[i+1] = stride[i] * dims[i];
    }
  }
  checkCUDNN(cudnnSetTensorNdDescriptor(m->inputTensor,
                                        CUDNN_DATA_FLOAT,
                                        DIM,
                                        dims,
                                        stride));
                                      
  checkCUDNN(cudnnCreateActivationDescriptor(&m->activation));
  checkCUDNN(cudnnSetActivationDescriptor(
    m->activation,
    CUDNN_ACTIVATION_TANH,
    CUDNN_NOT_PROPAGATE_NAN,
    1.0
  ));                                        
#endif
  return m;
}

template <int DIM>
void Tanh<DIM>::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  auto task_id = TANH_3D_INIT_TASK_ID;
  if (DIM == 3) {
    task_id = TANH_3D_INIT_TASK_ID;
  } else if (DIM == 2) {
    task_id = TANH_2D_INIT_TASK_ID;
  } else {
    printf("idim %d odim %d not supported", DIM, DIM);
  }
  IndexLauncher launcher(task_id, task_is,
    TaskArgument(this, sizeof(Tanh)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
      RegionRequirement(input_lps[0], 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
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



/*
  regions[0](I): input
  regions[1](O): output
*/  
template <int DIM>
void Tanh<DIM>::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  float alpha = 1.0f, beta = 0.0f;
  FFHandler handler = *((const FFHandler*) task->local_args);
  TanhMeta* m = new TanhMeta(handler);
  // TensorAccessorR<float, DIM> acc_input(
  //   regions[0], task->regions[0], FID_DATA, ctx, runtime);
  // TensorAccessorW<float, DIM> acc_output(
  //   regions[1], task->regions[1], FID_DATA, ctx, runtime,
  //   false/*readOutput*/);
  // assert(acc_input.rect.volume() == acc_output.rect.volume());
  // Rect<DIM> rect_input, rect_output;
  // rect_input = runtime->get_index_space_domain(ctx, task->regions[0].region.get_index_space());
  // rect_output = runtime->get_index_space_domain(ctx, task->regions[1].region.get_index_space());
  // const float *input_ptr = acc_input.ptr(rect_input.lo);
  // float *output_ptr = acc_output.ptr(rect_output.lo);


  Rect<DIM> rect_input_tensor = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const AccessorRO<float, DIM> acc_input_tensor(regions[0], FID_DATA);
  const float* input_ptr = acc_input_tensor.ptr(rect_input_tensor.lo);

  Rect<DIM> rect_output_tensor = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  const AccessorWO<float, DIM> acc_output_tensor(regions[1], FID_DATA);
  float* output_ptr = acc_output_tensor.ptr(rect_output_tensor.lo);


  // DOUBLE CHECK HANDLE TO PREVENT SEGMENTATION FAULT
  checkCUDA(cudnnActivationForward(
    m->handle.dnn,
    m->activation,
    &alpha, m->inputTensor, input_ptr,
    &beta, m->inputTensor, output_ptr
  ));
}

template <int DIM>
void Tanh<DIM>::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  auto task_id = TANH_3D_FWD_TASK_ID;
  if (DIM == 3) {
    task_id = TANH_3D_FWD_TASK_ID;
  } else if (DIM == 2) {
    task_id = TANH_2D_FWD_TASK_ID;
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
void Tanh<DIM>::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  float alpha = 1.0f, beta = 0.0f;
  FFHandler handler = *((const FFHandler*) task->local_args);
  TanhMeta* m = new TanhMeta(handler);
  Rect<DIM> rect_input_tensor = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  const AccessorRO<float, DIM> acc_input_tensor(regions[0], FID_DATA);
  const float* input_ptr = acc_input_tensor.ptr(rect_input_tensor.lo);

  Rect<DIM> rect_output_tensor = runtime->get_index_space_domain(
    ctx, task->regions[1].region.get_index_space());
  const AccessorRO<float, DIM> acc_output_tensor(regions[1], FID_DATA);
  const float* output_ptr = acc_output_tensor.ptr(rect_output_tensor.lo);

  Rect<DIM> rect_input_grad_tensor = runtime->get_index_space_domain(
    ctx, task->regions[2].region.get_index_space());
  const AccessorWO<float, DIM> acc_input_grad_tensor(regions[2], FID_DATA);
  float* input_grad_ptr = acc_input_grad_tensor.ptr(rect_input_grad_tensor.lo);

  Rect<DIM> rect_output_grad_tensor = runtime->get_index_space_domain(
    ctx, task->regions[3].region.get_index_space());
  const AccessorRO<float, DIM> acc_output_grad_tensor(regions[3], FID_DATA);
  const float* output_grad_ptr = acc_output_grad_tensor.ptr(rect_output_grad_tensor.lo);
  // DOUBLE CHECK HANDLE TO PREVENT SEGMENTATION FAULT
  checkCUDA(cudnnActivationBackward(
    m->handle.dnn,
    m->activation,
    &alpha, 
    m->inputTensor, output_ptr,
    m->inputTensor, output_grad_ptr,
    m->inputTensor, input_ptr,
    &beta, m->inputTensor, input_grad_ptr
  ));
}

template <int DIM>
void Tanh<DIM>::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<DIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<DIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  auto task_id = TANH_3D_BWD_TASK_ID;
  if (DIM == 3) {
    task_id = TANH_3D_BWD_TASK_ID;
  } else if (DIM == 2) {
    task_id = TANH_2D_BWD_TASK_ID;
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
        WRITE_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(2, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part_grad, 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}



template Tanh<3>::Tanh(FFModel& model,
  const std::string& pcname,
  const Tensor& _input);
template Tanh<2>::Tanh(FFModel& model,
  const std::string& pcname,
  const Tensor& _input);
template OpMeta* Tanh<3>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template OpMeta* Tanh<2>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Tanh<3>::init(const FFModel& ff);
template void Tanh<2>::init(const FFModel& ff);
template void Tanh<3>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Tanh<2>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Tanh<3>::forward(const FFModel& ff);
template void Tanh<2>::forward(const FFModel& ff);
template void Tanh<3>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Tanh<2>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Tanh<3>::backward(const FFModel& ff);
template void Tanh<2>::backward(const FFModel& ff);
template Tensor FFModel::tanh<3>(std::string name, const Tensor& input);
template Tensor FFModel::tanh<2>(std::string name, const Tensor& input);