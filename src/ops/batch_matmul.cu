// Copyright 2020 Facebook
#include "model.h"
#include "cuda_helper.h"
#include <iostream>

Tensor FFModel::batch_matmul(std::string name,
             const Tensor& input1, const Tensor& input2,
             const bool trans1,
             const bool trans2)
{
  BatchMatmul *bmm = new BatchMatmul(*this, name, input1, input2, trans1, trans2);
  layers.push_back(bmm);
  return bmm->output;
}


BatchMatmul::BatchMatmul(
  FFModel& model,
  const std::string& pcname,
  const Tensor& input1,
  const Tensor& input2,
  const bool trans1,
  const bool trans2
): Op(pcname, input1, input2), profiling(model.config.profiling){
  ArgumentMap argmap;
  // Retrive the task indexspace for the op
  task_is = model.get_or_create_task_is(pcname);
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  FieldSpace fs = model.config.field_space;
  /*
  input1 (m,k,d)
  input2 (n,k,d)
  output (n,m,d)
  */
  int d = input1.adim[2];
  int m = input1.adim[0];
  int n = input2.adim[0];
  int k = input1.adim[1];
  const int dims[] = {d,m,n};
  if (profiling){ 
    printf("batch_matmul inputs:\n");
    printf("input 1 shape d(%d) k(%d) m(%d)\n", d,k,m);
    printf("input 2 shape d(%d) k(%d) n(%d)\n", d,k,n);
  }
  transpose_1_flag = trans1;
  transpose_2_flag = trans2;
  transpose_1 = trans1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  transpose_2 = trans2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  // create 3-dimensional output tensor for this layer to hold the results
  output = model.create_tensor<3>(dims, pcname, DT_FLOAT);

  // Compute partition bound for input
  // TODO the input partition check can be refactored into a helper function
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  Rect<3> part_rect = domain;
  Rect<3> input1_rect = runtime->get_index_partition_color_space(
    ctx, input1.part.get_index_partition());
  if (input1_rect == part_rect) {
    input_lps[0] = input1.part;
    input_grad_lps[0] = input1.part_grad;
  } else {
    model.create_disjoint_partition<3>(
      input1,
      IndexSpaceT<3>(task_is),
      input_lps[0],
      input_grad_lps[0]
    );
  }
  Rect<3> input2_rect = runtime->get_index_partition_color_space(
    ctx, input2.part.get_index_partition());
  if (input2_rect == part_rect) {
    input_lps[1] = input2.part;
    input_grad_lps[1] = input2.part_grad;
  } else {
    model.create_disjoint_partition<3>(
      input2,
      IndexSpaceT<3>(task_is),
      input_lps[1],
      input_grad_lps[1]
    );
  }
  /*
  HACK: We zero-init the gradience of the output tensor inside the layer
  DISCUSSION AND QUESTION:
  It doesn't hurt to zero init the output gradients here, but is it the best way to do it?
  What's the performance downside? 
  What's the design downside?
  */
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  Domain output_grad_domain = runtime->get_index_partition_color_space(
    ctx, output.part_grad.get_index_partition());
  IndexSpace output_grad_task_is = model.get_or_create_task_is(output_grad_domain);
  IndexLauncher launcher(ZERO_INIT_TASK_ID, output_grad_task_is,
               TaskArgument(NULL, 0), argmap,
               Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
               FFConfig::get_hash_id(std::string("init output gradients")));
  launcher.add_region_requirement(
    RegionRequirement(output.part_grad, 0/*projection*/,
              WRITE_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(0, FID_DATA);
  runtime->execute_index_space(ctx, launcher);

}


void BatchMatmul::init(const FFModel& ff){
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // currently only support 3 dimensional batch matmul , outter dimension is sample dimension
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID, task_is,
    TaskArgument(this, sizeof(BatchMatmul)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
  RegionRequirement(output.part, 0/*projection id*/,
    WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
  RegionRequirement(inputs[0].part, 0/*projection id*/,
    READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
  RegionRequirement(inputs[1].part, 0/*projection id*/,
    READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(2, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}


/*
  regions[0](O): output
  regions[1](I): input1
  regions[2](I): input2
*/
OpMeta* BatchMatmul::init_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const BatchMatmul* bm = (BatchMatmul*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  TensorAccessorW<float, 3> acc_output(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    false/*readOutput*/);
  TensorAccessorR<float, 3> input1(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, 3> input2(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  /*
  input1 (m,k,d)
  input2 (n,k,d)
  output (n,m,d)
  */
  int k = input1.rect.hi[1] - input1.rect.lo[1] + 1;
  int m = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int n = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_stride_a = input1.rect.hi[2] - input1.rect.lo[2] + 1;
  int batch_stride_b = input2.rect.hi[2] - input2.rect.lo[2] + 1;
  int batch_stride_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  BatchMatmulMeta* bmm_meta = new BatchMatmulMeta(handle);
  if (bm->profiling){ 
    printf("init batch_matmul (input): batdh_dim(%d) k(%d) m(%d) n(%d)\n", batch_stride_a, k, m, n);
  }
  return bmm_meta;
}


/*
  regions[0](O): output
  regions[1](I): input1
  regions[2](I): input2
*/
void BatchMatmul::forward_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime
  )
{
  const BatchMatmul* bm = (BatchMatmul*) task->args;
  float alpha = 1.0f, beta = 0.0f;
  const BatchMatmulMeta* lm = *((BatchMatmulMeta**) task->local_args);
  const int batch_tensor_dim = 3;
  TensorAccessorW<float, batch_tensor_dim> acc_output(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    false/*readOutput*/);
  TensorAccessorR<float, batch_tensor_dim> acc_input1(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
  TensorAccessorR<float, batch_tensor_dim> acc_input2(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);
  /*
  shape d,k,m
  order d(2),k(1),m(0)
  axis  m,k,d
  index   2 1 0
  input1 (d,k,m)
  input2 (d,k,n)
  output (d,m,n)
  */
  int k = acc_input1.rect.hi[1] - acc_input1.rect.lo[1] + 1;
  int m = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int n = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_stride_a = acc_input1.rect.hi[2] - acc_input1.rect.lo[2] + 1;
  int batch_stride_b = acc_input2.rect.hi[2] - acc_input2.rect.lo[2] + 1;
  int batch_stride_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  if (bm->profiling){ 
    printf("k:%d m:%d n:%d batch_stride_a:%d batch_stride_b:%d batch_stride_c:%d\n", k, m,n,batch_stride_a, batch_stride_b, batch_stride_c);
    printf("cuBLAS initializing...\n");
  }
#ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(lm->handle.blas, stream));
  checkCUDNN(cudnnSetStream(lm->handle.dnn, stream));
#endif
  /*
    This scenario assumes the default input layout is 
          A (d,m,k) 
          B (d,k,n) 
          C (d,m,n)
    but the scenario we implement here is
          A (d,k,m) 
          B (d,k,n) 
          C (d,m,n)
    in order to map to default layout, we provide two flags (transposeA and tranposeB), in order to match the input shapes to default layout,
    we need to set transposeA=True and tranposeB=False
    So the forward and backward implementation are for transposeA=True & tranposeB=False
    other configurations are not implemented
  */
  if (bm->transpose_1_flag) {
    if (bm->transpose_2_flag) {
      // not implemented
      throw 255;
    }
    else{
      /*
      Leading dimension
           ||
           \/ 
      A (d,k,m) 
      B (d,k,n) 
      C (d,m,n)
      cuda representation:
      Matmul(B,A) = (d,m,n) = (d,k,n) * (d,m,k) 
      explain: becaise cuda is column major ordering, above is equivalent to the row major order 
      in most literatures
      (d,m,n) = (d,m,k) * (d,k,n)
      proof:
      in cuda 
      C^T = A^T*B^T
      in our scenario A is A^T, so
      C^T = A * B^T
      transpose C we get:
      C = B * A^T
      */
      checkCUDA(
        cublasSgemmStridedBatched(
          lm->handle.blas,
          CUBLAS_OP_N,
          CUBLAS_OP_T,
          n, m,  k,
          &alpha,
          acc_input2.ptr, n,
          k*n,
          acc_input1.ptr, m,
          m*k,
          &beta,
          acc_output.ptr, n,
          m*n,
          batch_stride_a)
      );

    }
  }
  else {
    if (bm->transpose_2_flag) {
      // not implemented
      throw 255;
    }
    else{
      // not implemented
      throw 255;
    }
  }
  if (bm->profiling){ 
    printf("input1 d:%d k:%d m:%d\n", batch_stride_a, k, m );
    print_tensor<3, float>(acc_input1.ptr, acc_input1.rect, "[BatchMatmul:forward:input1]");
    printf("input2 d:%d k:%d n:%d\n", batch_stride_a, k, n );
    print_tensor<3, float>(acc_input2.ptr, acc_input2.rect, "[BatchMatmul:forward:input2]");
    printf("output d:%d m:%d n:%d\n", batch_stride_a, m, n );
    print_tensor<3, float>(acc_output.ptr, acc_output.rect, "[BatchMatmul:forward:output]");
  }
}


void BatchMatmul::forward(const FFModel& ff){
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // currently only support 3 dimensional batch matmul , outter dimension is sample dimension
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(BATCHMATMUL_FWD_TASK_ID, task_is,
               TaskArgument(this, sizeof(BatchMatmul)), argmap,
               Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
               FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(output.part, 0/*projection id*/,
              WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < 2; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i+1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}


/*
  regions[0](O): output_grad
  regions[1](I): input1_grad
  regions[2](I): input2_grad
  regions[3](I): input1
  regions[4](I): input2
*/
void BatchMatmul::backward_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime
  )
{
  const BatchMatmul* bm = (BatchMatmul*) task->args;
  float alpha = 1.0f, beta = 0.0f;
  const BatchMatmulMeta* lm = *((BatchMatmulMeta**) task->local_args);
  const int batch_tensor_dim = 3;
  TensorAccessorR<float, batch_tensor_dim> acc_output_grad(
    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, batch_tensor_dim> acc_input1_grad(
    regions[1], task->regions[1], FID_DATA, ctx, runtime, false/*readOutput*/);

  TensorAccessorW<float, batch_tensor_dim> acc_input2_grad(
    regions[2], task->regions[2], FID_DATA, ctx, runtime, false/*readOutput*/);
  TensorAccessorR<float, batch_tensor_dim> acc_input1(
    regions[3], task->regions[3], FID_DATA, ctx, runtime);

  TensorAccessorR<float, batch_tensor_dim> acc_input2(
    regions[4], task->regions[4], FID_DATA, ctx, runtime);
  /*
  cuda representation:
  input1_grad = output_grad^T*input2
  (d,k,m) = (d,n,m) * (d,k,n)
  input2_grad = output*input1
  (d,k,n) = (d,m,n) * (d,k,m)
  for proof please check forward
  index   2 1 0
  input1 (d,k,m)
  input2 (d,k,n)
  output (d,m,n)
  */
  int k = acc_input1_grad.rect.hi[1] - acc_input1_grad.rect.lo[1] + 1;
  int m = acc_output_grad.rect.hi[1] - acc_output_grad.rect.lo[1] + 1;
  int n = acc_output_grad.rect.hi[0] - acc_output_grad.rect.lo[0] + 1;
  int batch_stride_a = acc_input1_grad.rect.hi[2] - acc_input1_grad.rect.lo[2] + 1;
  int batch_stride_b = acc_input2_grad.rect.hi[2] - acc_input2_grad.rect.lo[2] + 1;
  int batch_stride_c = acc_output_grad.rect.hi[2] - acc_output_grad.rect.lo[2] + 1;
  if (bm->profiling){ 
    printf("k:%d m:%d n:%d batch_stride_a:%d batch_stride_b:%d batch_stride_c:%d\n", k, m,n,batch_stride_a, batch_stride_b, batch_stride_c);
  }

  #ifndef DISABLE_LEGION_CUDA_HIJACK
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  checkCUDA(cublasSetStream(lm->handle.blas, stream));
  checkCUDNN(cudnnSetStream(lm->handle.dnn, stream));
  #endif
  if (bm->transpose_1_flag) {
    if (bm->transpose_2_flag) {
      // AB
      // not implemented
      throw 255;
    }
    else {
      // A'B
      // dA = BG', dB = AG
      // cuda (column major) representation:
      // m=n,n=m,k=k
      // input1 (d,k,m)
      // input2 (d,k,n)
      // output (d,m,n)
      // (d,k,m) = (d,n,m) * (d,k,n)
      // input1_grad = output_grad^T * input2
      checkCUDA(cublasSgemmStridedBatched(lm->handle.blas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    m,k,n,
                    &alpha,
                    acc_output_grad.ptr, n,
                    m*n,
                    acc_input2.ptr, n,
                    k*n,
                    &beta,
                    acc_input1_grad.ptr, m,
                    m*k,
                    batch_stride_a));
      // (d,k,n) = (d,m,n) * (d,k,m)
      // input2_grad = output_grad * input1
      checkCUDA(cublasSgemmStridedBatched(lm->handle.blas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n,k,m,
                    &alpha,
                    acc_output_grad.ptr, n,
                    m*n,
                    acc_input1.ptr, m,
                    m*k,
                    &beta,
                    acc_input2_grad.ptr, n,
                    k*n,
                    batch_stride_a));
    }
  } else {
    if (bm->transpose_2_flag) {
      // AB'
      // not implemented
      throw 255;

    }
    else {
      // AB
      // not implemented
      throw 255;
    }
  }

  if (bm->profiling){ 
    print_tensor<3, float>(acc_output_grad.ptr, acc_output_grad.rect, "[BatchMatmul:backward:acc_output_grad]");
    print_tensor<3, float>(acc_input1_grad.ptr, acc_input1_grad.rect, "[BatchMatmul:backward:input1_gard]");
    print_tensor<3, float>(acc_input2_grad.ptr, acc_input2_grad.rect, "[BatchMatmul:backward:input2_gard]");
  }

}


void BatchMatmul::backward(const FFModel& ff){
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // currently only support 3 dimensional batch matmul , outter dimension is sample dimension
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(BATCHMATMUL_BWD_TASK_ID, task_is,
              TaskArgument(this, sizeof(BatchMatmul)), argmap,
              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
              FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(output.part_grad, 0/*projection id*/,
              READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(0, FID_DATA);
  // input1 grad
  launcher.add_region_requirement(
            RegionRequirement(input_grad_lps[0], 0/*projection id*/,
            WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(1, FID_DATA);
  // input 2 grad
  launcher.add_region_requirement(
            RegionRequirement(input_grad_lps[1], 0/*projection id*/,
            WRITE_ONLY, EXCLUSIVE, inputs[1].region_grad));
  launcher.add_field(2, FID_DATA);
  // input1
  launcher.add_region_requirement(
            RegionRequirement(inputs[0].part, 0/*projection id*/,
            READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(3, FID_DATA);
  // input2
  launcher.add_region_requirement(
            RegionRequirement(inputs[1].part, 0/*projection id*/,
            READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(4, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

