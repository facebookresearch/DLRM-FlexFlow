#include "model.h"
#include "cuda_helper.h"

/*
 1. Copy forward and forward_task
 2. Make changes according to batch_matmul
 */


/*
 https://github.com/flexflow/FlexFlow/blob/d88e8b373d45e2227ba2700ca01060ea7267d633/src/ops/linear.cu#L271
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
*/
__host__
void BatchMatmul::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
    float alpha = 1.0f, beta = 0.0f;
    const Linear* linear = (Linear*) task->args;
    const LinearMeta* lm = *((LinearMeta**) task->local_args);
    TensorAccessorR<float, 2> acc_input(
                                        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorW<float, 2> acc_output(
          regions[1], task->regions[1], FID_DATA, ctx, runtime,
                                         false/*readOutput*/);
    TensorAccessorR<float, 2> acc_kernel(
                                         regions[2], task->regions[2], FID_DATA, ctx, runtime);
    
    /*
    Need confirmation on following sizes
    */
    int k = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
    int m = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
    int n = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
    int batch_count = acc_input.rect.hi[2] - acc_input.rect.hi[2] + 1;
    assert(acc_output.rect.volume() == batch_count * m * n);
    assert(acc_kernel.rect.volume() == batch_count * k * m);

    cudaEvent_t t_start, t_end;
    if (linear->profiling) {
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_end);
        cudaEventRecord(t_start);
    }
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDA(cublasSetStream(lm->handle.blas, stream));
    checkCUDA(
              cublasSgemmBatched(
                                  lm->handle.blas,
                                  CUBLAS_OP_T, CUBLAS_OP_N, (int)m, (int)n, (int)k,
                                  &alpha,
                                  acc_kernel.ptr, k,
                                  acc_input.ptr, k,
                                  &beta,
                                  acc_output.ptr, m
                                  (int)batchCount
                                 )
            );


    

    if (linear->activation != AC_MODE_NONE) {
        checkCUDNN(cudnnActivationForward(lm->handle.dnn, lm->actiDesc,
        &alpha, lm->outputTensor, acc_output.ptr,
        &beta, lm->outputTensor, acc_output.ptr));
    }
    if (linear->profiling) {
        cudaEventRecord(t_end);
        checkCUDA(cudaEventSynchronize(t_end));
        float elapsed = 0;
        checkCUDA(cudaEventElapsedTime(&elapsed, t_start, t_end));
        cudaEventDestroy(t_start);
        cudaEventDestroy(t_end);
        printf("Linear forward time = %.2lfms\n", elapsed);
        print_tensor<2, float>(acc_input.ptr, acc_input.rect, "[Linear:forward:input]");
        print_tensor<2, float>(acc_kernel.ptr, acc_kernel.rect, "[Linear:forward:kernel]");
        print_tensor<2, float>(acc_output.ptr, acc_output.rect, "[Linear:forward:output]");
        checkCUDA(cudaDeviceSynchronize());
    }
}

void BatchMatmul::backward(const FFModel& ff) {

}

BatchMatmul::BatchMatmul(FFModel& model,
               const std::string& pcname,
               const Tensor& _input,
               int out_dim,
               ActiMode _activation,
               bool use_bias,
               Initializer* kernel_initializer,
               Initializer* bias_initializer)
: Op(pcname, _input), activation(_activation),
  profiling(model.config.profiling)
{

  int tensor_dim = 3
  assert(_input.numDim == tensor_dim);
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<tensor_dim>(model.get_or_create_task_is(pcname));

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<tensor_dim> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int num_par_c = part_rect.hi[0] - part_rect.lo[0] + 1;
  int num_par_n = part_rect.hi[1] - part_rect.lo[1] + 1;
//  int in_dim = _input.adim[0];
//  int batch_size = _input.adim[1];
    /*
     Check the correctness of dimension sizes
     */
    int batch_count = _input.adim[2];
    int n = _input.adim[1];
    int k = _input.adim[0];
    int m = out_dim;
  {
      // HERE
//    const int dims[2] = {batch_size, out_dim};
    const int dims[3] = {batch_count, n, m};
    output = model.create_tensor<3>(dims, task_is, DT_FLOAT);
  }
  // Create kernel tensor
  {
      // HERE
//    const int dims[2] = {out_dim, in_dim};
      const int dims[3] = {batch_count, m, k};
    kernel = model.create_weight<3>(dims, task_is, DT_FLOAT, kernel_initializer);
  }
  // Compute partition bound for input
  Rect<3> input_rect = runtime->get_index_partition_color_space(
      ctx, inputs[0].part.get_index_partition());
    
    
    
    
    
  // !QUESTION!: what is num_par_c dimension
  // !QUESTION!: why we need replica tensor
  // Create replica tensor
  if (num_par_c > 1) {
//    const int dims[3] = {num_par_c, batch_size, in_dim};
    const int dims[4] = {num_par_c, batch_count, n, k};
    
      replica = model.create_replica<4>(dims, task_is, DT_FLOAT);
    {
      // !QUESTION!: how does transform work
      Rect<2> extent(Point<2>(0, 0), Point<2>(in_dim-1, batch_size/num_par_n-1));
      Transform<2, 2> transform;
      transform[0][0] = 0;
      transform[0][1] = 0;
      transform[1][0] = 0;
      transform[1][1] = batch_size/num_par_n;
      IndexPartition ip = runtime->create_partition_by_restriction(
          ctx, inputs[0].region.get_index_space(), task_is, transform, extent);
      input_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region, ip);
    }
    // !QUESTION! do we need backward
//    // Backward use the same ip as inputs[0]
//    input_grad_lps[0] = inputs[0].part_grad;
//    {
//      IndexSpaceT<2> input_task_is = IndexSpaceT<2>(model.get_or_create_task_is(input_rect));
//      const coord_t num_parts[2] = {input_rect.hi[0] - input_rect.lo[0] + 1,
//                                    input_rect.hi[1] - input_rect.lo[1] + 1};
//      Rect<3> extent(Point<3>(0, 0, 0),
//          Point<3>(in_dim/num_parts[0]-1, batch_size/num_parts[1]-1, num_par_c-1));
//      Transform<3, 2> transform;
//      for (int i = 0; i < 3; i++)
//        for (int j = 0; j < 2; j++)
//          transform[i][j] = 0;
//      transform[0][0] = in_dim / num_parts[0];
//      transform[1][1] = batch_size / num_parts[1];
//      IndexPartition ip = runtime->create_partition_by_restriction(
//          ctx, replica.region_grad.get_index_space(), input_task_is,
//          transform, extent);
//      assert(runtime->is_index_partition_disjoint(ctx, ip));
//      assert(runtime->is_index_partition_complete(ctx, ip));
//      // Note we use replica.part to save how to partition the replica
//      // to compute input_grad_lps
//      replica.part = runtime->get_logical_partition(
//          ctx, replica.region_grad, ip);
//    }
  } else {
    if (input_rect == part_rect) {
      input_lps[0] = inputs[0].part;
      input_grad_lps[0] = inputs[0].part_grad;
    } else {
      Rect<2> extent(Point<2>(0,0), Point<2>(in_dim-1,batch_size/num_par_n-1));
      Transform<2, 2> transform;
      transform[0][0] = 0;
      transform[0][1] = 0;
      transform[1][0] = 0;
      transform[1][1] = batch_size / num_par_n;
      IndexPartition ip = runtime->create_partition_by_restriction(
          ctx, inputs[0].region.get_index_space(), task_is, transform, extent);
      assert(runtime->is_index_partition_disjoint(ctx, ip));
      assert(runtime->is_index_partition_complete(ctx, ip));
      input_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region, ip);
      input_grad_lps[0] = runtime->get_logical_partition(
          ctx, inputs[0].region_grad, ip);
    }
  }
}


void BatchMatmul::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(BATCHMATMUL_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
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
  launcher.add_region_requirement(
      RegionRequirement(kernel.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, kernel.region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void BatchMatmul::init(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    FFHandler handle = ff.handlers[idx++];
    argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
  }
  IndexLauncher launcher(BATCHMATMUL_INIT_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Linear)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  //launcher.add_region_requirement(
  //    RegionRequirement(input_lps[0], 0/*projection id*/,
  //                      READ_ONLY, EXCLUSIVE, inputs[0].region));
  //launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(output.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(kernel.part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, kernel.region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  idx = 0;
  for (PointInRectIterator<2> it(rect); it(); it++) {
    meta[idx++] = fm.get_result<OpMeta*>(*it);
  }
}


/*
  regions[0](O): output
  regions[1](I): kernel
*/
OpMeta* Linear::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Linear* linear = (Linear*) task->args;
  FFHandler handle = *((const FFHandler*) task->local_args);
  //TensorAccessorR<float, 2> acc_input(
  //    regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, 2> acc_output(
      regions[0], task->regions[0], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  TensorAccessorR<float, 2> acc_kernel(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
    
  /*
  Need confirmation on following sizes
  */
  // int k = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int k = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int m = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int n = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int batch_count = acc_output.rect.hi[2] - acc_output.rect.hi[2] + 1;
    
    
  printf("init batch_matmul: k(%d) m(%d) n(%d), d(%d)\n",
      k, m, n, batch_count);
  LinearMeta* lm = new LinearMeta(handle);

  /*
  Need confirmation on following sizes
  */
    float *batched_dram_one_ptr[batch_count];
    for (int i = 0; i < batch_count; ++i) {
        float* dram_one_ptr = (float *) malloc(sizeof(float) * n);
        for (int i = 0; i < n; i++)
            dram_one_ptr[i] = 1.0f;
        batched_dram_one_ptr[i] = dram_one_ptr;
    }
    float* batched_fb_one_ptr;
    checkCUDA(cudaMalloc(&batched_fb_one_ptr, sizeof(float) * n * batch_count));
    checkCUDA(cudaMemcpy(batched_fb_one_ptr, batched_dram_one_ptr,
                       sizeof(float) * n * batch_count, cudaMemcpyHostToDevice));
    lm->one_ptr = (const float*) fb_one_ptr;
    if (linear->activation != AC_MODE_NONE) {
        cudnnActivationMode_t mode;
        switch (linear->activation) {
            case AC_MODE_RELU:
                mode = CUDNN_ACTIVATION_RELU;
                break;
            case AC_MODE_SIGMOID:
                mode = CUDNN_ACTIVATION_SIGMOID;
                break;
            default:
                // Unsupported activation mode
                assert(false);
    }
    checkCUDNN(cudnnCreateActivationDescriptor(&lm->actiDesc));
    checkCUDNN(cudnnSetActivationDescriptor(lm->actiDesc, mode,
                                            CUDNN_PROPAGATE_NAN, 0.0));
    checkCUDNN(cudnnCreateTensorDescriptor(&lm->outputTensor));
    /*
     TODO @charles Need to verify this
     */
     // !QUESTION!: how to describe tensor
    checkCUDNN(cudnnSetTensor4dDescriptor(lm->outputTensor,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                      batch_count, 1, m, n));
  }
  return m;
}
