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
    int dims[MAX_DIM], num_dim = input1.numDim;
    
    
    
    int tensor_obj_n_dim = 3
    
    Rect<tensor_obj_n_dim> part_rect = domain;
    output = model.create_tensor<tensor_obj_n_dim>(dims, IndexSpaceT<tensor_obj_n_dim>(task_is), DT_FLOAT);
    
    Rect<tensor_obj_n_dim> input1_rect = runtime->get_index_partition_color_space(
      ctx, input1.part.get_index_partition());
    if (input1_rect == part_rect) {
    input_lps[0] = input1.part;
    input_grad_lps[0] = input1.part_grad;
    } else {
     model.create_disjoint_partition<tensor_obj_n_dim>(input1,
         IndexSpaceT<tensor_obj_n_dim>(task_is), input_lps[0], input_grad_lps[0]);
    }
    
    Rect<tensor_obj_n_dim> input2_rect = runtime->get_index_partition_color_space(
      ctx, input2.part.get_index_partition());
    if (input2_rect == part_rect) {
    input_lps[1] = input2.part;
    input_grad_lps[1] = input2.part_grad;
    } else {
     model.create_disjoint_partition<tensor_obj_n_dim>(input2,
         IndexSpaceT<tensor_obj_n_dim>(task_is), input_lps[1], input_grad_lps[1]);
    }

    /*
    transformation
    output.adim
    output.pdim
    output.region
    outputpart
    output.part_grad
    */
}


void BatchMatmul::init(const FFmodel& ff){
    // do nothing here
}



/*
    !QUESTION!: confirm the region layout
  regions[0](O): output
  regions[1..numInputs](I): inputs
*/
void BatchMatmul::forward_task(
    const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx, Runtime *runtime
    ){
    const Concat* cc = (Concat*) task->args;
    
    int batch_tensor_dim = 3
    TensorAccessorW<float, batch_tensor_dim> acc_output(
        regions[0], task->regions[0], FID_DATA, ctx, runtime,
        false/*readOutput*/);
    output = accOutput.ptr;
    TensorAccessorR<float, batch_tensor_dim> acc_input(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    
    TensorAccessorR<float, batch_tensor_dim> acc_kernel(
        regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    




    // assert(regions.size() == batch_tensor_dim);
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

    // !QUESTION!: do i need this here
    checkCUDA(cudaDeviceSynchronize());
    
    // !QUESTION!: how do i return results
}


void BatchMatmul::forward(const FFModel& ff){
    ArgumentMap argmap;
    Context ctx = ff.config.lg_ctx;
    Runtime* runtime = ff.config.lg_hlr;
    
    
    
    
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
      launcher.add_field(i, FID_DATA);
    }
    runtime->execute_index_space(ctx, launcher);
}
