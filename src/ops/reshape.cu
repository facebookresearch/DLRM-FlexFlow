#pragma warning disable
#include "model.h"
#include "cuda_helper.h"

template <int IDIM, int ODIM>
Tensor FFModel::reshape(const Tensor& input, const int output_shape[])
{
  Reshape<IDIM,ODIM> *reshape = new Reshape<IDIM,ODIM>(*this, input, output_shape);
  layers.push_back(reshape);
  return reshape->outputs[0];
}


template <int IDIM, int ODIM>
Reshape<IDIM, ODIM>::Reshape(FFModel& model,
  const Tensor& _input,
  const int _output_shape[])
: Op(model, OP_RESHAPE, "Reshape_",  _input)
{
  outputs[0].numDim = ODIM;
  for (int i = 0; i < ODIM; i++)
    outputs[0].adim[i] = _output_shape[ODIM-1-i];
  numWeights = 0;
}

template <int IDIM, int ODIM>
Tensor Reshape<IDIM, ODIM>::init_inout(FFModel& model, const Tensor& _input)
{
  // TODO: This function is designed for support functional APIs
  // as used in PyTorch and Keras
  // TO BE IMPLEMENTED...
  assert(false);
  return Tensor();
}

template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::create_weights(FFModel& model)
{
  // Do nothing
}

template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::create_output_and_partition(FFModel& model)
{
  std::string pcname = name;
  task_is = IndexSpaceT<ODIM>(model.get_or_create_task_is(ODIM, pcname));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<ODIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_tasks = part_rect.volume();
  // number batches has to be divisible by partitions
  assert(inputs[0].adim[inputs[0].numDim-1] % num_tasks == 0);
  // Create output tensor
  int output_shape[ODIM];
  for (int i = 0; i < ODIM; i++)
    output_shape[i] = outputs[0].adim[ODIM-1-i];
  outputs[0] = model.create_tensor<ODIM>(output_shape, task_is, DT_FLOAT);
  model.create_data_parallel_partition_with_diff_dims<IDIM, ODIM>(
      inputs[0], (IndexSpaceT<ODIM>)task_is, input_lps[0], input_grad_lps[0]);
}

template <int IDIM, int ODIM>
OpMeta* Reshape<IDIM, ODIM>::init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  FFHandler handler = *((const FFHandler*) task->local_args);
  ReshapeMeta* m = new ReshapeMeta(handler);
  return m;
}

template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<ODIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<ODIM> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  auto task_id = RESHAPE_3_TO_2_INIT_TASK_ID;
  if (IDIM == 3 && ODIM == 2) {
    task_id = RESHAPE_3_TO_2_INIT_TASK_ID;
  } else if (IDIM == 2 && ODIM == 3) {
    task_id = RESHAPE_2_TO_3_INIT_TASK_ID;
  } else {
    printf("idim %d odim %d not supported", IDIM, ODIM);
  }
  IndexLauncher launcher(task_id, task_is,
    TaskArgument(this, sizeof(Reshape)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<ODIM> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

/*
  regions[0](I): input
  regions[1](O): output
*/  
template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::forward_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorR<float, IDIM> acc_input(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, ODIM> acc_output(
    regions[1], task->regions[1], FID_DATA, ctx, runtime,
    false/*readOutput*/);
  assert(acc_input.rect.volume() == acc_output.rect.volume());
  checkCUDA(cudaMemcpyAsync(acc_output.ptr, acc_input.ptr,
    acc_input.rect.volume() * sizeof(float),
    cudaMemcpyDeviceToDevice));
}

template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<ODIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<ODIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  auto task_id = RESHAPE_3_TO_2_FWD_TASK_ID;
  if (IDIM == 3 && ODIM == 2) {
    task_id = RESHAPE_3_TO_2_FWD_TASK_ID;
  } else if (IDIM == 2 && ODIM == 3) {
    task_id = RESHAPE_2_TO_3_FWD_TASK_ID;
  } else {
    printf("idim %d odim %d not supported", IDIM, ODIM);
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
      RegionRequirement(outputs[0].part /*3D->2D partitions*/, 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](O) : input_grad
  regions[1](I) : output_grad
*/
template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::backward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, IDIM> acc_input_grad(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    true/*readOutput*/);
  TensorAccessorR<float, ODIM> acc_output_grad(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());
  checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
    acc_input_grad.rect.volume() * sizeof(float),
    cudaMemcpyDeviceToDevice));
}

template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<ODIM> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<ODIM> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  auto task_id = RESHAPE_3_TO_2_BWD_TASK_ID;
  if (IDIM == 3 && ODIM == 2) {
    task_id = RESHAPE_3_TO_2_BWD_TASK_ID;
  } else if (IDIM == 2 && ODIM == 3) {
    task_id = RESHAPE_2_TO_3_BWD_TASK_ID;
  } else {
    printf("idim %d odim %d not supported", IDIM, ODIM);
  }
  IndexLauncher launcher(task_id, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
                      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

template <int IDIM, int ODIM>
void Reshape<IDIM, ODIM>::print_layer(const FFModel& ff)
{
}

template <int IDIM, int ODIM>
bool Reshape<IDIM, ODIM>::measure_compute_time(Simulator* sim,
                                  const ParallelConfig& pc,
                                  float& forward_time,
                                  float& backward_time)
{
  return false;
}

template Reshape<3,2>::Reshape(FFModel& model,
  const Tensor& _input,
  const int output_shape[]);
template Reshape<2,3>::Reshape(FFModel& model,
  const Tensor& _input,
  const int output_shape[]);
template OpMeta* Reshape<3,2>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template OpMeta* Reshape<2,3>::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Reshape<3,2>::init(const FFModel& ff);
template void Reshape<2,3>::init(const FFModel& ff);
template void Reshape<3,2>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Reshape<2,3>::forward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Reshape<3,2>::forward(const FFModel& ff);
template void Reshape<2,3>::forward(const FFModel& ff);
template void Reshape<3,2>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Reshape<2,3>::backward_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
template void Reshape<3,2>::backward(const FFModel& ff);
template void Reshape<2,3>::backward(const FFModel& ff);
template void Reshape<3,2>::print_layer(const FFModel& ff);
template void Reshape<2,3>::print_layer(const FFModel& ff);
template bool Reshape<3,2>::measure_compute_time(Simulator* sim,
  const ParallelConfig& pc, float& forward_time,
  float& backward_time);
template bool Reshape<2,3>::measure_compute_time(Simulator* sim,
  const ParallelConfig& pc, float& forward_time,
  float& backward_time);

template Tensor FFModel::reshape<3,2>(const Tensor& input, const int output_shape[]);
template Tensor FFModel::reshape<2,3>(const Tensor& input, const int output_shape[]);
