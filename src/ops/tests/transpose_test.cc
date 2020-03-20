#include "model.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#define MAX_DATASET_PATH_LEN 1023
#define  PRECISION 16
using namespace Legion;

LegionRuntime::Logger::Category log_app("transpose_test");

struct TransposeTestMeta {
  int m,k,d;
  TransposeTestMeta(int _m, int _k, int _d) {
      m = _m, k = _k, d = _d;
  }
};

struct ArgsConfig {
  char dataset_path[MAX_DATASET_PATH_LEN];
};

void initialize_tensor_from_file(const std::string file_path, Tensor label, const FFModel& ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  TaskLauncher launcher(
      INIT_TENSOR_FORM_FILE_CPU_TASK,
      TaskArgument(&args_config, sizeof(args_config)));
  // regions[0]: full_sparse_input
  launcher.add_region_requirement(
      RegionRequirement(label.region,
                        WRITE_ONLY, EXCLUSIVE, label.region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void initialize_tensor_gradient_from_file(const std::string file_path, Tensor label, const FFModel& ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  TaskLauncher launcher(
      INIT_TENSOR_FORM_FILE_CPU_TASK,
      TaskArgument(&args_config, sizeof(args_config)));
  launcher.add_region_requirement(
      RegionRequirement(
        label.region_grad,
        WRITE_ONLY, EXCLUSIVE, 
        label.region_grad,
        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void initialize_tensor_from_file_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime) {
  const AccessorWO<float, 3> acc_label_tensor(regions[0], FID_DATA);
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  Rect<3> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  float* label_tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
  std::fstream myfile(file_path, std::ios_base::in);
  float a;
  int i = 0;
  while (myfile >> a)
  {
    label_tensor_ptr[i] = a;
    i++;
    // printf("%.6lf ", (float)tensor_ptr[i]);
  }   
}

void dump_region_to_file(FFModel &ff, LogicalRegion &region, std::string file_path)
{
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  TaskLauncher launcher(DUMP_TENSOR_CPU_TASK, 
                        TaskArgument(&args_config, sizeof(args_config)));
  launcher.add_region_requirement(
    RegionRequirement(
      region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
  );
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void dump_tensor_task(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime)
{
  assert(task->regions.size() == 1);
  assert(regions.size() == 1);
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  const AccessorRO<float, 3> acc_tensor(regions[0], FID_DATA);
  Rect<3> rect_fb = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  assert(acc_tensor.accessor.is_dense_arbitrary(rect_fb));
  const float* tensor_ptr = acc_tensor.ptr(rect_fb.lo);
  std::ofstream myfile;
  myfile.open (file_path);
  for (size_t i = 0; i < rect_fb.volume(); ++i) {
    // printf("%.6lf ", (float)tensor_ptr[i]);
    myfile << std::fixed << std::setprecision(PRECISION) << (float)tensor_ptr[i] << " ";
  }
  myfile.close();
}

TransposeTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int m,k,d;
  myfile >> m >> k >> d;
  return TransposeTestMeta(m,k,d);
}

void register_custom_tasks()
{
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_FORM_FILE_CPU_TASK, "Load Label");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task>(
        registrar, "Load Label Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task>(
        registrar, "Compare Tensor Task");
  }
}

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  // std::cout<< "test framework launched" << std::endl;
  auto test_meta = get_test_meta("test_meta.txt");
  FFConfig ffConfig;
  // Parse input arguments
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
  }
  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.profiling = false;
  ffConfig.field_space = runtime->create_field_space(ctx);
  // create ff model object
  FFModel ff(ffConfig);
  // create input tensor
  Tensor dense_input;
  {
    const int dims[3] = {test_meta.d,test_meta.m,test_meta.k}; // target shape (d,m,k)
    dense_input = ff.create_tensor<3>(dims, "transpose", DT_FLOAT);
  }
  // build transpose layer
  Tensor ret = ff.transpose("transpose", dense_input);
  // load inputs tensors and output gradients tensors for testing
  auto input1_file_path = "test_input1.txt";
  auto output_grad_file_path = "test_output_grad.txt";
  initialize_tensor_from_file(input1_file_path, dense_input, ff);
  initialize_tensor_gradient_from_file(output_grad_file_path, ret, ff);
  // run forward and backward to produce results
  ff.init_layers();
  ff.forward();
  ff.backward();
  // dump results to file for python validation
  dump_region_to_file(ff, ret.region, "output.txt");
  dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt");
}


