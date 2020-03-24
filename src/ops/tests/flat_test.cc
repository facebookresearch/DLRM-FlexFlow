#include "model.h"
// #include "test_utils.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#define  PRECISION 16
#define MAX_DATASET_PATH_LEN 1023

using namespace Legion;

LegionRuntime::Logger::Category log_app("Flat_test");
struct ArgsConfig {
  char dataset_path[MAX_DATASET_PATH_LEN];
  char data_type[30];
  int num_dim;
};
void initialize_tensor_from_file(const std::string file_path, 
  Tensor label, 
  const FFModel& ff, 
  std::string data_type="float", 
  int num_dim=3);

void initialize_tensor_gradient_from_file(const std::string file_path, 
    Tensor label, 
    const FFModel& ff, 
    std::string data_type,  int num_dim) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  strcpy(args_config.data_type, data_type.c_str());
  if (num_dim == 2) {
    TaskLauncher launcher(
        INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 3) {
    TaskLauncher launcher(
        INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 4) {
    TaskLauncher launcher(
        INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else {
    throw 255;
  }

}


void initialize_tensor_from_file(const std::string file_path, 
    Tensor label, 
    const FFModel& ff, 
    std::string data_type,  int num_dim) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  strcpy(args_config.data_type, data_type.c_str());
  if (num_dim == 2) {
    TaskLauncher launcher(
        INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 3) {
    TaskLauncher launcher(
        INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 4) {
    TaskLauncher launcher(
        INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else {
    throw 255;
  }

}



void initialize_tensor_2d_from_file_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime) {
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  std::string data_type((const char*)args_config.data_type);
  Rect<2> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (data_type == "int") {
    const AccessorWO<int, 2> acc_label_tensor(regions[0], FID_DATA);
    int* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    int a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
    }   
  } else if (data_type == "float") {
    const AccessorWO<float, 2> acc_label_tensor(regions[0], FID_DATA);
    float* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    float a;
    int i = 0;
    while (myfile >> a)
    {
      // std::cout << a << std::endl;
      tensor_ptr[i] = a;
      i++;
    } 
  }
}


void initialize_tensor_3d_from_file_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime) {
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  std::string data_type((const char*)args_config.data_type);
  Rect<3> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (data_type == "int") {
    const AccessorWO<int, 3> acc_label_tensor(regions[0], FID_DATA);
    int* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    int a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
    }   
  } else if (data_type == "float") {
    const AccessorWO<float, 3> acc_label_tensor(regions[0], FID_DATA);
    float* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    float a;
    int i = 0;
    while (myfile >> a)
    {
      // std::cout << a << std::endl;
      tensor_ptr[i] = a;
      i++;
    } 
  }
}

void initialize_tensor_4d_from_file_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime) {
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  std::string data_type((const char*)args_config.data_type);
  Rect<4> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (data_type == "int") {
    const AccessorWO<int, 4> acc_label_tensor(regions[0], FID_DATA);
    int* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    int a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
    }   
  } else if (data_type == "float") {
    const AccessorWO<float, 4> acc_label_tensor(regions[0], FID_DATA);
    float* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    float a;
    int i = 0;
    while (myfile >> a)
    {
      // std::cout << a << std::endl;
      tensor_ptr[i] = a;
      i++;
    } 
  }
}


void dump_region_to_file(FFModel &ff, LogicalRegion &region, std::string file_path, int dims=4)
{
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  if (dims == 2) {
    TaskLauncher launcher(DUMP_TENSOR_2D_CPU_TASK, 
                          TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
      RegionRequirement(
        region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
    );
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (dims == 3) {
    TaskLauncher launcher(DUMP_TENSOR_3D_CPU_TASK, 
                          TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
      RegionRequirement(
        region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
    );
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);

  } else if (dims == 4) {
    TaskLauncher launcher(DUMP_TENSOR_4D_CPU_TASK, 
                          TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
      RegionRequirement(
        region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
    );
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);

  } else
  {
    throw 255;
  }
  

}

void dump_3d_tensor_task(const Task* task,
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

void dump_4d_tensor_task(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime)
{
  assert(task->regions.size() == 1);
  assert(regions.size() == 1);
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  const AccessorRO<float, 4> acc_tensor(regions[0], FID_DATA);
  Rect<4> rect_fb = runtime->get_index_space_domain(
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



void dump_2d_tensor_task(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime)
{
  assert(task->regions.size() == 1);
  assert(regions.size() == 1);
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  const AccessorRO<float, 2> acc_tensor(regions[0], FID_DATA);
  Rect<2> rect_fb = runtime->get_index_space_domain(
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











struct FlatTestMeta {
  int i_dim, o_dim;
  int* i_shape; 
  int* o_shape;
  FlatTestMeta(int _i_dim, int _o_dim, int* _i_shape, int* _o_shape) {
      i_dim = _i_dim;
      o_dim = _o_dim;
      i_shape = _i_shape;
      o_shape = _o_shape;
  }
};



FlatTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int b;
  std::vector<int> buffer;
  while (myfile >> b) 
  { 
      buffer.push_back(b); 
  } 
  int i_dim(buffer[0]), o_dim(buffer[1]);
  int* i_shape = new int[i_dim];
  int* o_shape = new int[o_dim];
  int offset = 2;
  for (int i = 0; i < i_dim; i++){
    i_shape[i] = buffer[i+offset];
  }
  offset += i_dim;
  for (int i = 0; i < o_dim; i++){
    o_shape[i] = buffer[i+offset];
  }
  // int m,k,d;
  // myfile >> m >> k >> d;
  return FlatTestMeta(i_dim, o_dim, i_shape, o_shape);
}

void register_custom_tasks()
{
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_2D_FROM_FILE_CPU_TASK, "Load 2d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_2d_from_file_task>(
        registrar, "Load 2d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_3D_FROM_FILE_CPU_TASK, "Load 3d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_3d_from_file_task>(
        registrar, "Load 3d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_4D_FROM_FILE_CPU_TASK, "Load 4d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_4d_from_file_task>(
        registrar, "Load 4d tensor Task");
  }

  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_2d_tensor_task>(
        registrar, "Compare Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_4d_tensor_task>(
        registrar, "Compare Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_3d_tensor_task>(
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
  Tensor dense_input;
#define input_dim 3
  const int i_dims[input_dim] = {
    test_meta.i_shape[0], 
    test_meta.i_shape[1], 
    test_meta.i_shape[2]
    // test_meta.i_shape[3]
  }; 
  // std::cout << test_meta.i_shape[0] << test_meta.i_shape[1] << test_meta.i_shape[2] << test_meta.i_shape[3] <<  std::endl;
  dense_input = ff.create_tensor<input_dim>(i_dims, "flat_3_in", DT_FLOAT);
  Tensor ret = ff.flat("flat_2_out", dense_input);
  auto input1_file_path = "test_input1.txt";
  auto output_grad_file_path = "test_output_grad.txt";
  initialize_tensor_from_file(input1_file_path, dense_input, ff, "float", 3);
  initialize_tensor_gradient_from_file(output_grad_file_path, ret, ff, "float", 2);
  // run forward and backward to produce results
  ff.init_layers();
  // forward
  ff.forward();
  dump_region_to_file(ff, ret.region, "output.txt", 2);

  ff.backward();
  dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt", 3);

  
  
}


