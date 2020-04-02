#include "model.h"
#include "test_utils.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
using namespace Legion;
LegionRuntime::Logger::Category log_app("dot_compressor_test");

struct DotCompressorTestMeta {
  int batch_size, i_dim, num_channels, projected_num_channels, dense_projection_i_dim;
  DotCompressorTestMeta(int _batch_size, int _i_dim, int _num_channels,
    int _projected_num_channels, int _dense_projection_i_dim) {
      batch_size = _batch_size, num_channels = _num_channels, 
        i_dim = _i_dim, projected_num_channels = _projected_num_channels,
        dense_projection_i_dim = _dense_projection_i_dim;
  }
};

DotCompressorTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int batch_size, i_dim, num_channels, projected_num_channels, dense_projection_i_dim;
  myfile >> batch_size >> i_dim >> num_channels >> projected_num_channels >> dense_projection_i_dim;
  return DotCompressorTestMeta(batch_size, i_dim, num_channels, projected_num_channels, dense_projection_i_dim);
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

  // create dense projection
  Tensor dense_projection;
  {
    const int dims[2] = {test_meta.batch_size, test_meta.i_dim}; // target shape (d,m,k)
    dense_input = ff.create_tensor<2>(dims, "transpose", DT_FLOAT);
  }

  // create embeddings
  int dense_embedding_channels = test_meta.num_channels / 2;
  int sparse_embedding_channels = test_meta.num_channels - dense_embedding_size;
  auto dense_embedding_file_path = "test_input2.txt";
  auto sparse_embedding_file_path = "test_input3.txt";
  Tensor dense_embeddings[dense_embedding_channels];
  Tensor sparse_embeddings[sparse_embedding_channels];
  for(int i = 0; i < dense_embedding_channels; i++) {
    const int dims[2] = {test_meta.batch_size, test_meta.i_dim};
    dense_embedding_channels[i] = ff.create_tensor<2>(dims, 
      "", DT_FLOAT);
    initialize_tensor_from_file(dense_embedding_file_path, 
      dense_embedding_channels[i], ff, "float", 2);
  }
  for(int i = 0; i < sparse_embedding_channels; i++) {
    const int dims[2] = {test_meta.batch_size, test_meta.i_dim};
    sparse_embedding_channels[i] = ff.create_tensor<2>(dims, 
      "", DT_FLOAT);
    initialize_tensor_from_file(sparse_embedding_file_path, 
      sparse_embedding_channels[i], ff, "float", 2);
  }


  // build transpose layer
  Tensor ret = ff.transpose("transpose", dense_input);
  // load inputs tensors and output gradients tensors for testing
  auto output_grad_file_path = "test_output_grad.txt";
  initialize_tensor_gradient_from_file(output_grad_file_path, ret, ff, "float", 3);
  // run forward and backward to produce results
  ff.init_layers();
  ff.forward();
  ff.backward();
  // dump results to file for python validation
  dump_region_to_file(ff, ret.region, "output.txt", 3);
  dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt", 3);
}


void register_custom_tasks()
{
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
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<2>>(
        registrar, "Compare Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<4>>(
        registrar, "Compare Tensor Task");
  }
  {      
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<3>>(
        registrar, "Compare Tensor Task");
  }
}
