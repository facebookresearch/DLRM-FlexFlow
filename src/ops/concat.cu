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

Tensor FFModel::concat(std::string name,
                       int n, const Tensor* tensors,
                       int axis)
{
  Concat *cat = new Concat(*this, name, n, tensors, axis);
  layers.push_back(cat);
  return cat->output;
}

Concat::Concat(FFModel& model,
               const std::string& pcname, 
               int _n, const Tensor* _tensors,
               int _axis)
 : Op(pcname, _n, _tensors), axis(_axis),
   profiling(model.config.profiling)
{
  // Retrive the task indexspace for the op
  task_is = model.get_or_create_task_is(inputs[0].numDim, pcname);

  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Domain domain = runtime->get_index_space_domain(ctx, task_is);
  FieldSpace fs = model.config.field_space;
  int dims[MAX_DIM], num_dim = inputs[0].numDim;
  assert(num_dim == domain.get_dim());
  for (int i = 0; i < num_dim; i++)
    dims[i] = inputs[0].adim[num_dim-1-i];
  // accumulate concatenate dimension
  for (int i = 1; i < numInputs; i++)
    for (int j = 0; j < num_dim; j++) {
      if (j != axis)
        assert(inputs[i].adim[num_dim-1-j] == dims[j]);
      else
        dims[j] += inputs[i].adim[num_dim-1-j];
    }
  //for (int i = 0; i < num_dim; i++)
  //  printf("concat: dim[%d] = %d\n", i, dims[i]);
  switch (domain.get_dim()) {
    // case 1:
    // {
    //   Rect<1> part_rect = domain;
    //   output = model.create_tensor<1>(dims, task_is, DT_FLOAT);
    //   for (int i = 0; i < numInputs; i++) {
    //     model.create_data_parallel_partition_with_diff_dims<1, 1>(
    //       inputs[i], IndexSpaceT<1>(task_is), input_lps[i], input_grad_lps[i]);
    //   }
    //   break;
    // }
    case 2:
    {
      Rect<2> part_rect = domain;
      output = model.create_tensor<2>(dims, task_is, DT_FLOAT);
      for (int i = 0; i < numInputs; i++) {
        model.create_data_parallel_partition_with_diff_dims<2, 2>(
          inputs[i], IndexSpaceT<2>(task_is), input_lps[i], input_grad_lps[i]);
      }
      break;
    }
    // case 3:
    // {
    //   Rect<3> part_rect = domain;
    //   output = model.create_tensor<3>(dims, task_is, DT_FLOAT);
    //   for (int i = 0; i < numInputs; i++) {
    //     model.create_data_parallel_partition_with_diff_dims<3, 3>(
    //       inputs[i], IndexSpaceT<3>(task_is), input_lps[i], input_grad_lps[i]);
    //   }
    //   break;
    // }
    // case 4:
    // {
    //   Rect<4> part_rect = domain;
    //   output = model.create_tensor<4>(dims, task_is, DT_FLOAT);
    //   for (int i = 0; i < numInputs; i++) {
    //     model.create_data_parallel_partition_with_diff_dims<4, 4>(
    //       inputs[i], IndexSpaceT<4>(task_is), input_lps[i], input_grad_lps[i]);
    //   }
    //   break;
    // }
    default:
    {
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
    }
  }
}

__host__
OpMeta* Concat::init_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  return NULL;
}

void Concat::init(const FFModel& ff)
{
}

__global__
void add_with_stride(float* output,
                     const float* input,
                     int num_blocks,
                     int output_blk_size,
                     int input_blk_size,
                     int input_volume)
{
  int min_blk_size = min(output_blk_size, input_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * min_blk_size)
  {
    int blk_idx = i / min_blk_size;
    int blk_offset = i % min_blk_size;
    int input_offset = blk_idx * input_blk_size + blk_offset;
    int output_offset = blk_idx * output_blk_size + blk_offset;
    if (input_offset < input_volume) {
      output[output_offset] += input[input_offset];
    }
  }
}


/*
        copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
            output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
        output += input_blk_sizes[i];
*/
__global__
void copy_with_stride(float* output,
                      const float* input,
                      int num_blocks,
                      int output_blk_size,
                      int input_blk_size,
                      int input_volume)
{
  int min_blk_size = min(output_blk_size, input_blk_size);
  CUDA_KERNEL_LOOP(i, num_blocks * min_blk_size)
  {
    int blk_idx = i / min_blk_size;
    int blk_offset = i % min_blk_size;
    int input_offset = blk_idx * input_blk_size + blk_offset;
    int output_offset = blk_idx * output_blk_size + blk_offset;
    if (input_offset < input_volume) {
      // printf("offset out, in, block offset,i, blk_idx, input, input_blk_size\t%d\t%d\t%d\t%d\t%d\t%.4f\t%d\n",
      // output_offset, 
      // input_offset,
      // blk_offset, 
      // i, 
      // blk_idx,
      // input[input_offset],
      // input_blk_size);
      output[output_offset] = input[input_offset];
    }
  }
}

/*
  regions[0](O): output
  regions[1..numInputs](I): inputs
*/
void Concat::forward_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  // Note that our internal axis index ordering is opposite to other frameworks
  int axis = cc->output.numDim - 1 - cc->axis;
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  float *output;
  const float *inputs[MAX_NUM_INPUTS];
  int num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  for (int d = 0; d < cc->output.numDim; d++) {
    if (d <= axis)
      output_blk_size *= cc->output.adim[d];
    else
      num_blocks *= cc->output.adim[d];
  }
  for (int i = 0; i < cc->numInputs; i++) {
    input_blk_sizes[i] = 1;
    // each input has same block size
    for (int d = 0; d <= axis; d++)
      input_blk_sizes[i] *= cc->inputs[i].adim[d];
  }
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(domain.get_dim() == cc->output.numDim);
  switch (domain.get_dim()) {
    // case 1:
    // {
    //   TensorAccessorW<float, 1> accOutput(
    //       regions[0], task->regions[0], FID_DATA, ctx, runtime,
    //       false/*readOutput*/);
    //   output = accOutput.ptr;
    //   for (int i = 0; i < cc->numInputs; i++) {
    //     TensorAccessorR<float, 1> accInput(
    //         regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    //     inputs[i] = accInput.ptr;
    //   }
    //   break;
    // }
    case 2:
    {
      TensorAccessorW<float, 2> accOutput(
          regions[0], task->regions[0], FID_DATA, ctx, runtime,
          false/*readOutput*/);
      output = accOutput.ptr;

      
      for (int i = 0; i < cc->numInputs; i++) {
        TensorAccessorR<float, 2> accInput(
            regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
        inputs[i] = accInput.ptr;
        if (cc->profiling){
          printf("input volume %d\n", accInput.rect.volume());
          std::ostringstream stringStream;
          stringStream << "[Concat:forward:input" << i << "]";
          print_tensor<2, float>(inputs[i], accInput.rect, stringStream.str().c_str());
          printf("output = %x num_blocks=%d output_blk_size=%d input_blk_size[%d]=%d\n",
          output, num_blocks, output_blk_size, i, input_blk_sizes[i]);

        }
        copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
            output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i],
            accInput.rect.volume());
        output += input_blk_sizes[i];
      }
      checkCUDA(cudaDeviceSynchronize());
      if (cc->profiling) {
        printf("output_blk_size=%zu\n", output_blk_size);
        print_tensor<2, float>(accOutput.ptr, accOutput.rect, "[Concat:forward:output]");
      }
      return ;
    }
    // case 3:
    // {
    //   TensorAccessorW<float, 3> accOutput(
    //       regions[0], task->regions[0], FID_DATA, ctx, runtime,
    //       false/*readOutput*/);
    //   output = accOutput.ptr;
    //   for (int i = 0; i < cc->numInputs; i++) {
    //     TensorAccessorR<float, 3> accInput(
    //         regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    //     inputs[i] = accInput.ptr;
    //   }
    //   break;
    // }
    // case 4:
    // {
    //   TensorAccessorW<float, 4> accOutput(
    //       regions[0], task->regions[0], FID_DATA, ctx, runtime,
    //       false/*readOutput*/);
    //   output = accOutput.ptr;
    //   for (int i = 0; i < cc->numInputs; i++) {
    //     TensorAccessorR<float, 4> accInput(
    //         regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime);
    //     inputs[i] = accInput.ptr;
    //   }
    //   break;
    // }
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }
  // for (int i = 0; i < cc->numInputs; i++) {
  //   copy_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
  //       output, inputs[i], num_blocks, output_blk_size, input_blk_sizes[i]);
  //   printf("output = %x num_blocks=%d output_blk_size=%d input_blk_size[%d]=%d\n",
  //         output, num_blocks, output_blk_size, i, input_blk_sizes[i]);
  //   output += input_blk_sizes[i];
  // }
  // checkCUDA(cudaDeviceSynchronize());
  // if (cc->profiling) {
  //   printf("output_blk_size=%zu\n", output_blk_size);
  //   for (int i = 0; i < cc->numInputs; i++) {
  //     Rect<2> input_rec(Point<2>(0, 0), Point<2>(input_blk_sizes[i]-1, domain.get_volume() / output_blk_size - 1));
  //     std::ostringstream stringStream;
  //     stringStream << "[Concat:forward:input" << i << "]";
  //     print_tensor<2, float>(inputs[i], input_rec, stringStream.str().c_str());

  //   }
  //   Rect<2> rect(Point<2>(0, 0), Point<2>(output_blk_size-1, domain.get_volume() / output_blk_size - 1));
  //   print_tensor<2, float>(output - output_blk_size, rect, "[Concat:forward:output]");
  // }
}

void Concat::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(CONCAT_FWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Concat)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(output.part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, output.region));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_lps[i], 0/*projection id*/,
        READ_ONLY, EXCLUSIVE, inputs[i].region));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): output_grad
  regions[1..numInputs](O): input_grad
*/
void Concat::backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime)
{
  const Concat* cc = (Concat*) task->args;
  // Note that our internal axis index ordering is opposite to other frameworks
  int axis = cc->output.numDim - 1 - cc->axis;
  assert(regions.size() == cc->numInputs + 1);
  assert(task->regions.size() == cc->numInputs + 1);
  const float *output_grad;
  float *input_grads[MAX_NUM_INPUTS];
  int num_blocks = 1, output_blk_size = 1, input_blk_sizes[MAX_NUM_INPUTS];
  for (int d = 0; d < cc->output.numDim; d++) {
    if (d <= axis)
      output_blk_size *= cc->output.adim[d];
    else
      num_blocks *= cc->output.adim[d];
  }
  for (int i = 0; i < cc->numInputs; i++) {
    input_blk_sizes[i] = 1;
    for (int d = 0; d <= axis; d++)
      input_blk_sizes[i] *= cc->inputs[i].adim[d];
  }
  assert(cc->numInputs <= MAX_NUM_INPUTS);
  Domain domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(domain.get_dim() == cc->output.numDim);

  switch (domain.get_dim()) {
    // case 1:
    // {
    //   TensorAccessorR<float, 1> accOutputGrad(
    //       regions[0], task->regions[0], FID_DATA, ctx, runtime);
    //   output_grad = accOutputGrad.ptr;
    //   for (int i = 0; i < cc->numInputs; i++) {
    //     TensorAccessorW<float, 1> accInputGrad(
    //         regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime,
    //         false/*readOutput*/);
    //     input_grads[i] = accInputGrad.ptr;
    //   }
    //   break;
    // }
    case 2:
    {
      TensorAccessorR<float, 2> accOutputGrad(
          regions[0], task->regions[0], FID_DATA, ctx, runtime);
      output_grad = accOutputGrad.ptr;
      for (int i = 0; i < cc->numInputs; i++) {
        TensorAccessorW<float, 2> accInputGrad(
            regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime,
            false/*readOutput*/);
        input_grads[i] = accInputGrad.ptr;
        add_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
            input_grads[i], 
            output_grad, 
            num_blocks, 
            input_blk_sizes[i], 
            output_blk_size,
            accInputGrad.rect.volume());
        output_grad += input_blk_sizes[i];
        if (cc->profiling){
          printf("input volume %d\n", accInputGrad.rect.volume());
          std::ostringstream stringStream;
          stringStream << "[Concat:backward:input_grad" << i << "]";
          print_tensor<2, float>(input_grads[i], accInputGrad.rect, stringStream.str().c_str());
        }
      }
      checkCUDA(cudaDeviceSynchronize());
      if (cc->profiling) {
        int batch_size = domain.get_volume() / output_blk_size;
        Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size - 1));
        Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1, batch_size - 1));
        print_tensor<2, float>(output_grad - output_blk_size, output_rect, "[Concat:backward:output]");
      }
      return ;
    }
    // case 3:
    // {
    //   TensorAccessorR<float, 3> accOutputGrad(
    //       regions[0], task->regions[0], FID_DATA, ctx, runtime);
    //   output_grad = accOutputGrad.ptr;
    //   for (int i = 0; i < cc->numInputs; i++) {
    //     TensorAccessorW<float, 3> accInputGrad(
    //         regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime,
    //         false/*readOutput*/);
    //     input_grads[i] = accInputGrad.ptr;
    //   }
    //   break;
    // }
    // case 4:
    // {
    //   TensorAccessorR<float, 4> accOutputGrad(
    //       regions[0], task->regions[0], FID_DATA, ctx, runtime);
    //   output_grad = accOutputGrad.ptr;
    //   for (int i = 0; i < cc->numInputs; i++) {
    //     TensorAccessorW<float, 4> accInputGrad(
    //         regions[i+1], task->regions[i+1], FID_DATA, ctx, runtime,
    //         false/*readOutput*/);
    //     input_grads[i] = accInputGrad.ptr;
    //   }
    //   break;
    // }
    default:
      fprintf(stderr, "Unsupported concat dimension number");
      assert(false);
  }
  // for (int i = 0; i < cc->numInputs; i++) {
  //   add_with_stride<<<GET_BLOCKS(input_blk_sizes[i]*num_blocks), CUDA_NUM_THREADS>>>(
  //       input_grads[i], output_grad, num_blocks, input_blk_sizes[i], output_blk_size);
  //   output_grad += input_blk_sizes[i];
  // }
  // checkCUDA(cudaDeviceSynchronize());
  // if (cc->profiling) {
  //   int batch_size = domain.get_volume() / output_blk_size;
  //   Rect<2> output_rect(Point<2>(0, 0), Point<2>(output_blk_size-1, batch_size - 1));
  //   Rect<2> input_rect(Point<2>(0, 0), Point<2>(input_blk_sizes[0]-1, batch_size - 1));
  //   print_tensor<2, float>(output_grad - output_blk_size, output_rect, "[Concat:backward:output]");
  //   // print_tensor<2, float>(input_grads[0], input_rect, "[Concat:backward:input0]");
  // }
}

void Concat::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexLauncher launcher(CONCAT_BWD_TASK_ID, task_is,
    TaskArgument(this, sizeof(Concat)), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
    RegionRequirement(output.part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(0, FID_DATA);
  for (int i = 0; i < numInputs; i++) {
    launcher.add_region_requirement(
      RegionRequirement(input_grad_lps[i], 0/*projection id*/,
        WRITE_ONLY, EXCLUSIVE, inputs[i].region_grad));
    launcher.add_field(i + 1, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

