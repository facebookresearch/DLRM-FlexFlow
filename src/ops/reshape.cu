#pragma warning disable
#include "model.h"
#include "cuda_helper.h"


Tensor FFModel::reshape(std::string name, const Tensor& input, const int output_shape[], const int out_dim)
{
  if (input.numDim == 2 && out_dim == 3) {
    // Reshape2to3 *reshape = new Reshape2to3(*this, name, input, output_shape);
    // layers.push_back(reshape);
    // return reshape->output;
  }
  else if (input.numDim == 3 && out_dim == 2) {
    Reshape3to2 *reshape = new Reshape3to2(*this, name, input, output_shape);
    layers.push_back(reshape);
    return reshape->output;
  }
  throw 255;
}


Reshape3to2::Reshape3to2(FFModel& model,
  const std::string _pcname,
  const Tensor& _input,
  const int output_shape[])
: Op(_pcname, _input)
{
  pcname = _pcname;
  int i_volumn = 1;
  int o_volumn = 1;
  for (int i = 0; i < IDIM; i++) {
    i_volumn *= _input.adim[i];
  }
  for (int i = 0; i < ODIM; i++) {
    o_volumn *= output_shape[i];    
  }
  std::cout << "CP 2" << std::endl;
  assert(i_volumn == o_volumn);
  std::cout << "CP 2" << std::endl;
    
  task_is = IndexSpaceT<2>(model.get_or_create_task_is(2, pcname));
  // Context ctx = model.config.lg_ctx;
  // Runtime* runtime = model.config.lg_hlr;
  // Create output tensor
  {
    output = model.create_tensor<2>(output_shape, task_is, DT_FLOAT);
  }
  std::cout << "CP 2" << std::endl;
  model.create_data_parallel_partition_with_diff_dims<3, 2>(
    _input, task_is, input_lps[0], input_grad_lps[0]);
}



void Reshape3to2::init(const FFModel& ff)
{

}

/*
regions[0](I): input
regions[1](O): output
*/  
void Reshape3to2::forward_task(const Task *task,
               const std::vector<PhysicalRegion> &regions,
               Context ctx, Runtime *runtime)
{
assert(regions.size() == 2);
assert(task->regions.size() == 2);
TensorAccessorR<float, 3> acc_input(
regions[0], task->regions[0], FID_DATA, ctx, runtime);
TensorAccessorW<float, 2> acc_output(
regions[1], task->regions[1], FID_DATA, ctx, runtime,
false/*readOutput*/);

assert(acc_input.rect.volume() == acc_output.rect.volume());
checkCUDA(cudaMemcpyAsync(acc_output.ptr, acc_input.ptr,
                   acc_input.rect.volume() * sizeof(float),
                   cudaMemcpyDeviceToDevice));
}

void Reshape3to2::forward(const FFModel& ff)
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
  IndexLauncher launcher(RESHAPE_3_TO_2_FWD_TASK_ID, task_is,
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
regions[0](O) : input_grad
regions[1](I) : output_grad
*/
void Reshape3to2::backward_task(const Task *task,
                const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  TensorAccessorW<float, 3> acc_input_grad(
  regions[0], task->regions[0], FID_DATA, ctx, runtime,
  true/*readOutput*/);
  TensorAccessorR<float, 2> acc_output_grad(
  regions[1], task->regions[1], FID_DATA, ctx, runtime);
  assert(acc_input_grad.rect.volume() == acc_output_grad.rect.volume());

  checkCUDA(cudaMemcpyAsync(acc_input_grad.ptr, acc_output_grad.ptr,
                    acc_input_grad.rect.volume() * sizeof(float),
                    cudaMemcpyDeviceToDevice));
}

void Reshape3to2::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  IndexSpaceT<3> task_is = IndexSpaceT<3>(inputs[0].region.get_index_space());
  Rect<3> rect = runtime->get_index_space_domain(ctx, task_is);
  int idx = 0;
  for (PointInRectIterator<3> it(rect); it(); it++) {
    OpMeta* mp = meta[idx++];
    argmap.set_point(*it, TaskArgument(&mp, sizeof(OpMeta*)));
  }
  IndexLauncher launcher(RESHAPE_3_TO_2_BWD_TASK_ID, task_is,
    TaskArgument(NULL, 0), argmap,
    Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
    FFConfig::get_hash_id(std::string(name)));
  launcher.add_region_requirement(
  RegionRequirement(input_grad_lps[0], 0/*projection id*/,
                READ_WRITE, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
  RegionRequirement(output.part_grad, 0/*projection id*/,
                READ_ONLY, EXCLUSIVE, output.region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

