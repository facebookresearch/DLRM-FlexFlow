/* Copyright 2017 Stanford, NVIDIA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
): Op(pcname, input1, input2){
    ArgumentMap argmap;
    // Retrive the task indexspace for the op
    task_is = model.get_or_create_task_is(pcname);
    Context ctx = model.config.lg_ctx;
    Runtime* runtime = model.config.lg_hlr;
    Domain domain = runtime->get_index_space_domain(ctx, task_is);
    FieldSpace fs = model.config.field_space;
    // because input A is d,k,m
    const int dims[] = {input2.adim[0], input1.adim[0], input1.adim[2]};

    printf("trans A %d, trans B %d", trans1, trans2);
    transpose_1_flag = trans1;
    transpose_2_flag = trans2;
    transpose_1 = trans1 ? CUBLAS_OP_T : CUBLAS_OP_N;
    transpose_2 = trans2 ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int tensor_obj_n_dim = 3;

    Rect<tensor_obj_n_dim> part_rect = domain;
    output = model.create_tensor<tensor_obj_n_dim>(dims, "batch_matmul", DT_FLOAT);

    Rect<tensor_obj_n_dim> input1_rect = runtime->get_index_partition_color_space(
      ctx, input1.part.get_index_partition());
    if (input1_rect == part_rect) {
        input_lps[0] = input1.part;
        input_grad_lps[0] = input1.part_grad;
    } else {
        model.create_disjoint_partition<tensor_obj_n_dim>(input1,
        IndexSpaceT<3>(task_is), input_lps[0], input_grad_lps[0]);
    }

    Rect<tensor_obj_n_dim> input2_rect = runtime->get_index_partition_color_space(
      ctx, input2.part.get_index_partition());
    if (input2_rect == part_rect) {
    input_lps[1] = input2.part;
    input_grad_lps[1] = input2.part_grad;
    } else {
     model.create_disjoint_partition<tensor_obj_n_dim>(input2,
         IndexSpaceT<3>(task_is), input_lps[1], input_grad_lps[1]);
    }




    // initialize the output gradients temporarily , we dont have to do this once we connect the layer to a loss layer

    // currently only support 3 dimensional batch matmul , outter dimension is sample dimension
    Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
    int idx = 0;
    for (PointInRectIterator<3> it(rect); it(); it++) {
        OpMeta* mp = meta[idx++];
        argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
        // FFHandler handle = ff.handlers[idx++];
        // argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
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
    FFHandler handle = *((const FFHandler*) task->local_args);
    //TensorAccessorR<float, 2> acc_input(
    //    regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorW<float, 3> acc_output(
    regions[0], task->regions[0], FID_DATA, ctx, runtime,
    false/*readOutput*/);
    TensorAccessorR<float, 3> input1(
    regions[1], task->regions[1], FID_DATA, ctx, runtime);
    TensorAccessorR<float, 3> input2(
    regions[2], task->regions[2], FID_DATA, ctx, runtime);


    /*
    NEED TO VERIFY THE SHAPE,
    input1 (d,k,m)
    input2 (d,k,n)
    output (d,m,n)
    */
    int k = input1.rect.hi[0] - input1.rect.lo[0] + 1;
    int m = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
    int n = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
    int batch_stride_a = input1.rect.hi[2] - input1.rect.lo[2] + 1;
    int batch_stride_b = input2.rect.hi[2] - input2.rect.lo[2] + 1;
    int batch_stride_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;



    BatchMatmulMeta* bmm_meta = new BatchMatmulMeta(handle);
    printf("init batch_matmul (input): batdh_dim(%d) k(%d) m(%d) n(%d)\n", batch_stride_a, k, m, n);
    checkCUDNN(cudnnCreateTensorDescriptor(&bmm_meta->outputTensor));
    checkCUDNN(cudnnSetTensor4dDescriptor(bmm_meta->outputTensor,
                        CUDNN_TENSOR_NCHW,
                        CUDNN_DATA_FLOAT,
                        batch_stride_a, 1, m, n));
    return bmm_meta;
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
                         TaskArgument(this, sizeof(Concat)), argmap,
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

    /*
    IMPLEMENT 0,0 and 1,0 transpose scenario according to this
    scuba query
    https://our.internmc.facebook.com/intern/scuba/query/?dataset=caffe2_operator_stats&drillstate=%7B%22cols%22%3A[]%2C%22derivedCols%22%3A[%7B%22name%22%3A%22cpu_time_us%22%2C%22sql%22%3A%22cpu_time_ns%2F1000%22%2C%22type%22%3A%22Numeric%22%7D]%2C%22mappedCols%22%3A[]%2C%22enumCols%22%3A[]%2C%22return_remainder%22%3Afalse%2C%22hideEmptyColumns%22%3Afalse%2C%22start%22%3A%221581245391%22%2C%22samplingRatio%22%3A%221%22%2C%22compare%22%3A%22%22%2C%22hide_sample_cols%22%3Afalse%2C%22minBucketSamples%22%3A%22%22%2C%22dimensions%22%3A[%22operator_type%22%2C%22input_dims%22%2C%22output_dims%22%2C%22net_pos_int%22%2C%22arguments%22]%2C%22cellOverlay%22%3A%22None%22%2C%22metric%22%3A%22sum%22%2C%22top%22%3A%22200%22%2C%22timezone%22%3A%22America%2FLos_Angeles%22%2C%22end%22%3A%221581435008%22%2C%22aggregateList%22%3A[]%2C%22param_dimensions%22%3A[]%2C%22modifiers%22%3A[]%2C%22order%22%3A%22cpu_time_us%22%2C%22order_desc%22%3Atrue%2C%22filterMode%22%3A%22DEFAULT%22%2C%22constraints%22%3A[[%7B%22column%22%3A%22model_id%22%2C%22op%22%3A%22eq%22%2C%22value%22%3A[%22[%5C%22167229611%5C%22]%22]%7D%2C%7B%22column%22%3A%22operator_type%22%2C%22op%22%3A%22eq%22%2C%22value%22%3A[%22[%5C%22BatchMatMul%5C%22]%22]%7D]]%2C%22c_constraints%22%3A[[]]%2C%22b_constraints%22%3A[[]]%2C%22metrik_view_params%22%3A%7B%7D%7D&view=Table&selector=%23u_fetchstream_3_n&setRelative=true&height=474px&width=1000px&normalized=1581537964&pool=uber
    */
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

    int k = acc_input1_grad.rect.hi[0] - acc_input1_grad.rect.lo[0] + 1;
    int m = acc_output_grad.rect.hi[0] - acc_output_grad.rect.lo[0] + 1;
    int n = acc_input1_grad.rect.hi[1] - acc_input1_grad.rect.lo[1] + 1;
    int batch_stride_a = acc_input1_grad.rect.hi[2] - acc_input1_grad.rect.lo[2] + 1;
    int batch_stride_b = acc_input2_grad.rect.hi[2] - acc_input2_grad.rect.lo[2] + 1;
    int batch_stride_c = acc_output_grad.rect.hi[2] - acc_output_grad.rect.lo[2] + 1;
    printf("k:%d m:%d n:%d batch_stride_a:%d batch_stride_b:%d batch_stride_c:%d\n", k, m,n,batch_stride_a, batch_stride_b, batch_stride_c);
    printf("trans A %d, trans B %d", bm->transpose_1_flag, bm->transpose_2_flag);
    printf("cuBLAS initializing...\n");

    #ifndef DISABLE_LEGION_CUDA_HIJACK
    cudaStream_t stream;
    checkCUDA(cudaStreamCreate(&stream));
    checkCUDA(cublasSetStream(lm->handle.blas, stream));
    checkCUDNN(cudnnSetStream(lm->handle.dnn, stream));
    #endif

    // if (bm->transpose_1_flag) {
        // if (bm->transpose_2_flag) {
    if (true) {
        if (false) {
            // A'B':
            // dA = B'G', dB = G'A'
            // not implemented
            printf("A'B'");
            throw;
        }
        else {
            // A'B:
            // dA = BG', dB = AG
            printf("cuBLAS batch_matmul input1 grad job running...\n");
            checkCUDA(cublasSgemmStridedBatched(lm->handle.blas,
                                        CUBLAS_OP_N, CUBLAS_OP_T,
                                        k,m,n,
                                        &alpha,
                                        acc_input2.ptr, k,
                                        k*n,
                                        acc_output_grad.ptr, m,
                                        m*n,
                                        &beta,
                                        acc_input1_grad.ptr, k,
                                        m*k,
                                        batch_stride_a));
            printf("done!\n");
            printf("cuBLAS batch_matmul input2 grad job running...\n");
            checkCUDA(cublasSgemmStridedBatched(lm->handle.blas,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        k,n,m,
                                        &alpha,
                                        acc_input1.ptr, k,
                                        m*k,
                                        acc_output_grad.ptr, m,
                                        m*n,
                                        &beta,
                                        acc_input2_grad.ptr, k,
                                        k*n,
                                        batch_stride_a));
            printf("done!\n");
        }
    } else {
        if (bm->transpose_2_flag) {
            // AB':
            // dA = GB, dB = G'A
            // not implemented
            printf("AB'");
            throw;

        }
        else {
            // AB:
            // dA = GB', dB = A'G
            // not implemented
            printf("AB");
            throw;
        }
    }




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
    /*
    cublas function takes inputs and output in shape
    op ( A [ i ] ) m × k , op ( B [ i ] ) k × n and C [ i ] m × n
    so make sure op() outputs the correct shape,
    for example if A is in shape (m,k), then we set transposeA=false
    if A is in shape(k,m), we need to set transposeA=true in order to get
    op(A) in shape (m,k)
    */
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


    int k = acc_input1.rect.hi[0] - acc_input1.rect.lo[0] + 1;
    int m = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
    int n = acc_input1.rect.hi[1] - acc_input1.rect.lo[1] + 1;
    int batch_stride_a = acc_input1.rect.hi[2] - acc_input1.rect.lo[2] + 1;
    int batch_stride_b = acc_input2.rect.hi[2] - acc_input2.rect.lo[2] + 1;
    int batch_stride_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
    printf("k:%d m:%d n:%d batch_stride_a:%d batch_stride_b:%d batch_stride_c:%d\n", k, m,n,batch_stride_a, batch_stride_b, batch_stride_c);
    printf("cuBLAS initializing...\n");
    /*
        BUG HERE: SEGMENTATION FAULT
    */
    #ifndef DISABLE_LEGION_CUDA_HIJACK
        cudaStream_t stream;
        checkCUDA(cudaStreamCreate(&stream));
        checkCUDA(cublasSetStream(lm->handle.blas, stream));
        checkCUDNN(cudnnSetStream(lm->handle.dnn, stream));
    #endif
    printf("cuBLAS job running...\n");
    /*
    Figure out the trans_a and trans_b
    CUBLAS_OP_T, CUBLAS_OP_N,
    relationship

    assume OP(a) has shape m,k
    OP(b) has shape k,n
    OP(c) has shaoe m,n
    */
    checkCUDA(cublasSgemmStridedBatched(lm->handle.blas,
        bm->transpose_1,
        bm->transpose_2,
                                m, n, k,
                                &alpha,
                                acc_input1.ptr, k,
                                m*k,
                                acc_input2.ptr, k,
                                k*n,
                                &beta,
                                acc_output.ptr, m,
                                m*n,
                                batch_stride_a));
    printf("cuBLAS job done!\n");

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
        // FFHandler handle = ff.handlers[idx++];
        // argmap.set_point(*it, TaskArgument(&handle, sizeof(FFHandler)));
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
