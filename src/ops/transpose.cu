// Copyright 2020 Facebook
#include "model.h"
#include "cuda_helper.h"
#include <iostream>

Tensor FFModel::transpose(const Tensor& input)
{
  Transpose *trans = new Transpose(*this, input);
  layers.push_back(trans);
  return trans->outputs[0];
}

Transpose::Transpose(FFModel& model,
                     const Tensor& _input)
: Op(model, OP_TRANSPOSE, "Transpose_", _input), profiling(model.config.profiling)
{
}

Tensor Transpose::init_inout(FFModel& model, const Tensor& _input)
{
  // TODO: This function is designed for support functional APIs
  // as used in PyTorch and Keras
  // TO BE IMPLEMENTED...
  assert(false);
  return Tensor();
}

void Transpose::create_weights(FFModel& model)
{
  // Do nothing
}

void Transpose::create_output_and_partition(FFModel& model)
{
  // Retrive the task indexspace for the op
  std::string pcname = name;
  task_is = IndexSpaceT<3>(model.get_or_create_task_is(3, pcname));
  {
    int k = inputs[0].adim[0];
    int m = inputs[0].adim[1];
    int d = inputs[0].adim[2];
    const int dims[] = {d,k,m};
    outputs[0] = model.create_tensor<3>(dims, (IndexSpaceT<3>)task_is, DT_FLOAT);
    outputs[0].owner_op = this;
    outputs[0].owner_idx = 0;
  }

  model.create_data_parallel_partition_with_diff_dims<3, 3>(
    inputs[0], IndexSpaceT<3>(task_is), input_lps[0], input_grad_lps[0]);
}

void Transpose::init(const FFModel& ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // currently only support 3 dimensional transpose , outter dimension is sample dimension
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(TRANSPOSE_INIT_TASK_ID, task_is,
    TaskArgument(this, sizeof(Transpose)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
  RegionRequirement(outputs[0].part, 0/*projection id*/,
    WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
  RegionRequirement(inputs[0].part, 0/*projection id*/,
    READ_WRITE, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);
  
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}

OpMeta* Transpose::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Transpose* bm = (Transpose*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorW<float, 3> acc_output(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    false/*readOutput*/);
  TensorAccessorR<float, 3> input1(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  
  /*
  input1 (k,m,d)
  output (m,k,d)
  */
  int k = input1.rect.hi[0] - input1.rect.lo[0] + 1;
  int m = input1.rect.hi[1] - input1.rect.lo[1] + 1;
  int batch_stride_a = input1.rect.hi[2] - input1.rect.lo[2] + 1;
  int batch_stride_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  TransposeMeta* bmm_meta = new TransposeMeta(handle);
  if (bm->profiling){ 
    printf("init transpose (input): batdh_dim(%d) k(%d) m(%d) \n", batch_stride_a, k, m);
  }
  return bmm_meta;
}

void Transpose::forward(const FFModel& ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // currently only support 3 dimensional transpose , outter dimension is sample dimension
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(TRANSPOSE_FWD_TASK_ID, task_is,
    TaskArgument(this, sizeof(Transpose)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(inputs[0].part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0].region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


/*
  regions[0](I): input
  regions[1](O): output
*/
void Transpose::forward_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime) {
  const Transpose* bm = (Transpose*) task->args;
  float alpha = 1.0f, beta = 0.0f;
  const TransposeMeta* lm = *((TransposeMeta**) task->local_args);
  const int batch_tensor_dim = 3;
  TensorAccessorR<float, batch_tensor_dim> acc_input(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, batch_tensor_dim> acc_output(
    regions[1], task->regions[1], FID_DATA, ctx, runtime,
    false/*readOutput*/);
  /*
  shape d,m,k
  order d(2),m(1),k(0)
  axis    k,m,d
  index   2 1 0
  input1 (d,m,k)
  output (d,k,m)
  */
  int k = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int m = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int batch_stride_a = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int batch_stride_b = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  if (bm->profiling){ 
    printf("k:%d m:%d batch_stride_input:%d batch_stride_output:%d\n", k, m, batch_stride_a, batch_stride_b);
    printf("cuBLAS initializing...\n");
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(lm->handle.blas, stream));
  checkCUDNN(cudnnSetStream(lm->handle.dnn, stream));
#endif
  for(int batch_count = 0; batch_count < batch_stride_a; batch_count++) {
    int batch_stride = m*k;
    int offset = batch_count * batch_stride;
    checkCUDA(
      cublasSgeam(
        lm->handle.blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N, /*although we are not using this but still have to pass in correct shape*/
        m,k,
        &alpha,
        acc_input.ptr+offset, k,
        &beta,
        acc_input.ptr+offset, m, /*although we are not using this but still have to pass in correct shape*/
        acc_output.ptr+offset, m
      )
    );
  }
  if (bm->profiling){ 
    printf("input1 d:%d k:%d m:%d\n", batch_stride_a, k, m );
    print_tensor<3, float>(acc_input.ptr, acc_input.rect, "[Transpose:forward:input]");
    print_tensor<3, float>(acc_output.ptr, acc_output.rect, "[Transpose:forward:output]");
  }
}

void Transpose::backward(const FFModel& ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // currently only support 3 dimensional transpose , outter dimension is sample dimension
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(TRANSPOSE_BWD_TASK_ID, task_is,
    TaskArgument(this, sizeof(Transpose)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(inputs[0].part_grad, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0].part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Transpose::backward_task(
                        const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime
                        ){
  const Transpose* bm = (Transpose*) task->args;
  float alpha = 1.0f, beta = 0.0f;
  const TransposeMeta* lm = *((TransposeMeta**) task->local_args);
  const int batch_tensor_dim = 3;
  TensorAccessorW<float, batch_tensor_dim> acc_input(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    false/*readOutput*/);
  TensorAccessorR<float, batch_tensor_dim> acc_output(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  /*
  shape d,m,k
  order d(2),m(1),k(0)
  axis    k,m,d
  index   2 1 0
  input1 (d,m,k)
  output (d,k,m)
  */
  int k = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int m = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int batch_stride_a = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int batch_stride_b = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  if (bm->profiling){ 
    printf("k:%d m:%d batch_stride_input:%d batch_stride_output:%d\n", k, m, batch_stride_a, batch_stride_b);
    printf("cuBLAS initializing...\n");
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(lm->handle.blas, stream));
  checkCUDNN(cudnnSetStream(lm->handle.dnn, stream));
#endif
  for(int batch_count = 0; batch_count < batch_stride_a; batch_count++) {
    int batch_stride = m*k;
    int offset = batch_count * batch_stride;
    checkCUDA(
      cublasSgeam(
        lm->handle.blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        k,m,
        &alpha,
        acc_output.ptr+offset, m,
        &beta,
        acc_output.ptr+offset, k,
        acc_input.ptr+offset, k
      )
    );
  }
  if (bm->profiling){ 
    printf("input1 d:%d k:%d m:%d\n", batch_stride_a, k, m );
    print_tensor<3, float>(acc_input.ptr, acc_input.rect, "[Transpose:backward:input]");
    print_tensor<3, float>(acc_output.ptr, acc_output.rect, "[Transpose:backward:output]");
  }
}

void Transpose::print_layer(const FFModel& model)
{}

bool Transpose::measure_compute_time(Simulator* sim,
                                     const ParallelConfig& pc,
                                     float& forward_time,
                                     float& backward_time)
{
  return false;
}

