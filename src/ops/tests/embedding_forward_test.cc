#include "test_utils.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace Legion;

LegionRuntime::Logger::Category log_app("embedding_test");

struct EmbeddingTestMeta {
  int m,k,n,nnz;
  EmbeddingTestMeta(int _m, int _k, int _n, int _nnz) {
      m = _m, k = _k, n = _n, nnz=_nnz;
  }
};




EmbeddingTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int m,k,n, nnz;
  myfile >> m >> k >> n >> nnz;
  return EmbeddingTestMeta(m,k,n,nnz);
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
  int batch_size=test_meta.n; 
  int o_dim=test_meta.m;
  int i_dim=test_meta.k; 
  int nnz=test_meta.nnz;
  /*
  Sparse encoding:
  input shape {batch size, number non zero elements} 
  output shape {batch size, output dimension} 
  kernel shape {output dimension, input dimension}  
  */
  Tensor input;
  {
    const int dims[2] = {batch_size, nnz};
    input = ff.create_tensor<2>(dims, "embedding1", DT_INT64);
  }
  Tensor kernel;
  {
    // create weights placeholder 
    const int dims[2] = {o_dim, i_dim};
    kernel = ff.create_tensor<2>(dims, "embedding1", DT_FLOAT);
  }
  // initialize kernel (embedding) from file
  auto embedding_file_path = "test_kernel1.txt";  
  initialize_tensor_from_file(embedding_file_path, kernel, ff, "float", 2);
  
  Tensor ret = ff.embedding("embedding1", input, i_dim, o_dim, AGGR_MODE_SUM, kernel);

  // load inputs tensors and output gradients tensors for testing
  auto input1_file_path = "test_input1.txt";
  // looks like input is empty  , debug this @charles
  initialize_tensor_from_file(input1_file_path, input, ff, "int", 2);
  // run forward and backward to produce results
  ff.init_layers();
  ff.forward();

  // dump results to file for python validation
  dump_region_to_file(ff, ret.region, "output.txt");
}


