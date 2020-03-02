#include "model.h"
#include "hdf5.h"
#include <sstream>
#include <fstream>
#include <iostream>
#define MAX_DATASET_PATH_LEN 1023
#define MAX_NUM_SAMPLES 65536
using namespace Legion;

LegionRuntime::Logger::Category log_app("DLRM");

std::vector<float> read_numbers_from_file(const std::string file_path) {
    std::fstream myfile(file_path, std::ios_base::in);
    std::vector<float> buffer;
    float a;
    while (myfile >> a)
    {
        buffer.push_back(a);
    }
    return buffer;
}

struct BMMTestMeta {
    int m,k,n,d;
    BMMTestMeta(int _m, int _k, int _n, int _d) {
        m = _m, k = _k, n = _n, d = _d;
    }
};


struct ArgsConfig {
  char dataset_path[MAX_DATASET_PATH_LEN];
};

void InitializeTensorFromFile(const std::string file_path, Tensor label, const FFModel& ff) {
    Context ctx = ff.config.lg_ctx;
    Runtime* runtime = ff.config.lg_hlr;

    ArgsConfig args_config;
    strcpy(args_config.dataset_path, file_path.c_str());
    TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
      TaskArgument(&args_config, sizeof(args_config)));
  // regions[0]: full_sparse_input
  launcher.add_region_requirement(
      RegionRequirement(label.region,
                        WRITE_ONLY, EXCLUSIVE, label.region,
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

  auto label_value = read_numbers_from_file(file_path);
  for (size_t i = 0; i < rect_label_tensor.volume(); i++) {
    label_tensor_ptr[i] = label_value[i];
  }
      
}


bool is_equal(FFModel &ff, Tensor &t1, Tensor &t2)
{
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  TaskLauncher launcher(COMPARE_TENSOR_TASK, TaskArgument(NULL, 0));
  launcher.add_region_requirement(
    RegionRequirement(
      t1.region, READ_ONLY, EXCLUSIVE, t1.region, MAP_TO_ZC_MEMORY)
  );
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(
      t2.region, READ_ONLY, EXCLUSIVE, t2.region, MAP_TO_ZC_MEMORY)
  );
  launcher.add_field(1, FID_DATA);

  bool result = runtime->execute_task(ctx, launcher).get_result<bool>();
  return result;
}

bool compare_tensor_task(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime)
{
  assert(task->regions.size() == 2);
  assert(regions.size() == 2);

  const AccessorRO<float, 3> t1(regions[0], FID_DATA);
  Rect<3> rect_fb = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());

  const AccessorRO<float, 3> t2(regions[1], FID_DATA);
  Rect<3> rect_fb2 = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  assert(t1.accessor.is_dense_arbitrary(rect_fb));
  assert(t2.accessor.is_dense_arbitrary(rect_fb2));

  const float* t1_ptr = t1.ptr(rect_fb.lo);
  const float* t2_ptr = t2.ptr(rect_fb2.lo);
  float eplison = 0.000001;

  for (size_t i = 0; i < rect_fb.volume(); ++i) {
    if (t1_ptr[i] - t2_ptr[i] > eplison || t1_ptr[i] - t2_ptr[i] < -eplison ) {
      return false;
    }
  }
  return true;
}

BMMTestMeta get_test_meta(const std::string file_path) {
    std::fstream myfile(file_path, std::ios_base::in);
    int m,k,n,d;
    myfile >> m >> k >> n >> d;
    return BMMTestMeta(m,k,n,d);
}

void register_custom_tasks()
{
    // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Label");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task>(
        registrar, "Load Label Task");
  }
  {      
    TaskVariantRegistrar registrar(COMPARE_TENSOR_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<bool, compare_tensor_task>(
        registrar, "Compare Tensor Task");
  }
}

// ===================== Batch matmul

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{

  // int m = 265;
  // int k = 64;
  // int n = 15;
  // int d = 145;

  // simple problem for testing and debugging


    std::cout<< "test framework launched" << std::endl;
    auto input1_data = read_numbers_from_file("test_input.txt");
    auto input2_data = read_numbers_from_file("test_input.txt");
    
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
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);


  Tensor dense_input1;
  {


    const int dims[3] = {test_meta.d,test_meta.k,test_meta.m}; // target shape (d,k,m)
    // sadly we have to pass batch_matmul 3-dimensional stretegy in this way for now to handle 3 dimensional tensor
    dense_input1 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  Tensor dense_input2;
  {

    const int dims[3] = {test_meta.d,test_meta.k,test_meta.n}; // shape (n,k,d)
    // sadly we have to pass batch_matmul 3-dimensional stretegy in this way for now to handle 3 dimensional tensor
    dense_input2 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }

  Tensor label;
  {
    const int dims[3] = {test_meta.d,test_meta.m,test_meta.n}; // shape (n,m,d)
    // sadly we have to pass batch_matmul 3-dimensional stretegy in this way for now to handle 3 dimensional tensor
    label = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }


  Tensor batch_matmul_ret = ff.batch_matmul("batch_matmul", dense_input1, dense_input2, true, false);


auto output_file_path = "test_output.txt";
InitializeTensorFromFile(output_file_path, label, ff);
auto input1_file_path = "test_input1.txt";
auto input2_file_path = "test_input2.txt";
InitializeTensorFromFile(input1_file_path, dense_input1, ff);
InitializeTensorFromFile(input2_file_path, dense_input2, ff);


  ff.init_layers();
  ff.forward();
  ff.backward();
  bool ret = is_equal(ff, batch_matmul_ret, label);
  std::cout << "result:" << ret << std::endl;
  assert(ret == true);

}


