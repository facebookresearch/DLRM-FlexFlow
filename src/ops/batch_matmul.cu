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
                       const Tensor& input1, const Tensor& input2)
{
  BatchMatmul *bmm = new BatchMatmul(*this, name, input1, input2);
  layers.push_back(bmm);
  return bmm->output;
}


BatchMatmul::BatchMatmul(
    FFModel& model,
    const std::string& pcname,
    const Tensor& input1,
    const Tensor& input2
): Op(pcname, input1, input2){
    // Retrive the task indexspace for the op
    task_is = model.get_or_create_task_is(pcname);

    Context ctx = model.config.lg_ctx;
    Runtime* runtime = model.config.lg_hlr;
    Domain domain = runtime->get_index_space_domain(ctx, task_is);
    FieldSpace fs = model.config.field_space;
    const int dims[] = {input1.adim[2], input1.adim[1], input2.adim[0]};


    const int tensor_obj_n_dim = 3;

    Rect<tensor_obj_n_dim> part_rect = domain;
    output = model.create_tensor<tensor_obj_n_dim>(dims, IndexSpaceT<tensor_obj_n_dim>(task_is), DT_FLOAT);


    // initialize output tensor
    // Initializer* initializer = new ZeroInitializer();
    // initializer->init(ctx, runtime, &output);


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
    // to be implemented - change the shape
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

    float alpha = 1.0f, beta = 0.0f;
    /*
    SEGMENTATION FAULT HERE
    */
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
    checkCUDA(cublasSgemmStridedBatched(lm->handle.blas,
                                CUBLAS_OP_T, CUBLAS_OP_N,
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
                          WRITE_ONLY, EXCLUSIVE, output.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    for (int i = 0; i < 2; i++) {
      launcher.add_region_requirement(
          RegionRequirement(input_lps[i], 0/*projection id*/,
            READ_ONLY, EXCLUSIVE, inputs[i].region,
                            MAP_TO_ZC_MEMORY));
      launcher.add_field(i+1, FID_DATA);
    }
    runtime->execute_index_space(ctx, launcher);
}
