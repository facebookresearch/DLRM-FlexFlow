#include "model.h"
#include "test_utils.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#define  PRECISION 16
using namespace Legion;
LegionRuntime::Logger::Category log_app("sigmoid_ce_test");

struct TestMeta {
  int i_dim, o_dim;
  int* i_shape; 
  int* o_shape;
  TestMeta(int _i_dim, int _o_dim, int* _i_shape, int* _o_shape) {
      i_dim = _i_dim;
      o_dim = _o_dim;
      i_shape = _i_shape;
      o_shape = _o_shape;
  }
};

TestMeta get_test_meta(const std::string file_path) {
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

  return TestMeta(i_dim, o_dim, i_shape, o_shape);
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
  ffConfig.profiling = true;
  ffConfig.field_space = runtime->create_field_space(ctx);
  // create ff model object
  FFModel ff(ffConfig);
  Tensor dense_input;
  int* i_dims = new int[255];
  int* o_dims = new int[255];
  if (test_meta.i_dim == 2) {
#define input_dim 2
    for(int i=0; i<input_dim; ++i) {
      i_dims[i] = test_meta.i_shape[i];
    }
  }  
  else {
    printf("i_dim %d o_dim %d not supported\n" , test_meta.i_dim, test_meta.o_dim);
    throw 255;
  }

  if (test_meta.o_dim == 2) {
#define output_dim 2
    for(int i=0; i<output_dim; ++i) {

      o_dims[i] = test_meta.o_shape[i];
      std::cout << "o dim here" << o_dims[i] << "  " << test_meta.o_shape[i] << " o dim in meta" << test_meta.o_dim;
    }
  }  
  else {
    printf("i_dim %d o_dim %d not supported\n" , test_meta.i_dim, test_meta.o_dim);
    throw 255;
  }


  const int weight_shape[] = {i_dims[1], o_dims[1]};

  int seed = std::rand();
  Initializer* kernel_initializer = new GlorotUniform(seed);
  Initializer* bias_initializer = new ZeroInitializer();
  dense_input = ff.create_tensor<input_dim>(i_dims, "", DT_FLOAT);
  IndexSpace task_is = IndexSpaceT<2>(ff.get_or_create_task_is(input_dim, ""));
  Tensor weights = ff.create_linear_weight<input_dim>(weight_shape, (IndexSpaceT<2>)task_is, DT_FLOAT, kernel_initializer);
  // checked this already, where to initialize this doesn't matter
  auto input1_file_path = "test_input1.txt";
  auto linear_kernal_path = "test_kernel1.txt";
  initialize_tensor_from_file(input1_file_path, dense_input, ff, "float", input_dim);
  initialize_tensor_from_file(linear_kernal_path, weights, ff, "float", input_dim);




  
  Tensor linear_out = ff.dense(
    "linear",
    dense_input,
    test_meta.o_shape[output_dim-1],
    AC_MODE_NONE, 
    true,
    kernel_initializer, bias_initializer, 
    &weights, NULL
  );

  auto label_file_path = "test_label1.txt";
  Tensor labels = ff.create_tensor<output_dim>(test_meta.o_shape, "", DT_FLOAT);
  initialize_tensor_from_file(label_file_path, labels, ff, "float", output_dim);
  ff.sigmoid_cross_entropy_loss<output_dim>("", linear_out, labels);

  ff.optimizer = new SGDOptimizer(&ff, 0.01f, 0.0f);
  // run forward and backward to produce results
  ff.init_layers();
  // forward
  ff.forward();
  int epochs = 1;
  for (int i = 0; i < epochs; i++) {
    ff.backward();
    ff.update();
  }
  

  // dump_region_to_file(ff, dense_projection.region, "dump.txt", 2);
  dump_region_to_file(ff, ff.parameters[0].tensor.region, "kernel_updated1.txt", 2);
#undef input_dim
#undef output_dim
}




void register_custom_tasks()
{
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_1D_FROM_FILE_CPU_TASK, "Load 1d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<1>>(
        registrar, "Load 1d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_2D_FROM_FILE_CPU_TASK, "Load 2d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<2>>(
        registrar, "Load 2d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_3D_FROM_FILE_CPU_TASK, "Load 3d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<3>>(
        registrar, "Load 3d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_4D_FROM_FILE_CPU_TASK, "Load 4d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<4>>(
        registrar, "Load 4d tensor Task");
  }

  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_1D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<1>>(
        registrar, "Dump Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<2>>(
        registrar, "Dump Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<4>>(
        registrar, "Dump Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<3>>(
        registrar, "Dump Tensor Task");
  }
}