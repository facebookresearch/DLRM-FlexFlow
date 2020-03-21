#include "test_utils.h"

void initialize_tensor_from_file(const std::string file_path, 
    Tensor label, 
    const FFModel& ff, 
    std::string data_type, 
    int num_dim) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  strcpy(args_config.data_type, data_type.c_str());
  printf("lkasdjfiogawejgioawdgas\n");
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

ghaliuwehfuiwaeraewr
// todo here, this printing doesn't show
void initialize_tensor_2d_from_file_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime) {
#define NUM_DIM 2
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  std::string data_type((const char*)args_config.data_type);
  Rect<NUM_DIM> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (data_type == "int") {
    const AccessorWO<int, NUM_DIM> acc_label_tensor(regions[0], FID_DATA);
    int* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    int a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
      printf("%d ", (int)tensor_ptr[i]);
    }   
  } else if (data_type == "float") {
    const AccessorWO<float, NUM_DIM> acc_label_tensor(regions[0], FID_DATA);
    float* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
      std::fstream myfile(file_path, std::ios_base::in);
      float a;
      int i = 0;
      while (myfile >> a)
      {
        tensor_ptr[i] = a;
        i++;
        printf("%.6lf ", (float)tensor_ptr[i]);
      } 
  }
#undef NUM_DIM
}




void initialize_tensor_3d_from_file_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime) {
#define NUM_DIM 3
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  std::string data_type((const char*)args_config.data_type);
  Rect<NUM_DIM> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (data_type == "int") {
    const AccessorWO<int, NUM_DIM> acc_label_tensor(regions[0], FID_DATA);
    int* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    int a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
      printf("%d ", (int)tensor_ptr[i]);
    }   
  } else if (data_type == "float") {
    const AccessorWO<float, NUM_DIM> acc_label_tensor(regions[0], FID_DATA);
    float* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
      std::fstream myfile(file_path, std::ios_base::in);
      float a;
      int i = 0;
      while (myfile >> a)
      {
        tensor_ptr[i] = a;
        i++;
        printf("%.6lf ", (float)tensor_ptr[i]);
      } 
  }
#undef NUM_DIM
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

