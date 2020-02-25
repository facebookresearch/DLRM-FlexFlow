#include "model.h"
#include "hdf5.h"
#include <sstream>
#include <fstream>
#include <iostream>

#define MAX_NUM_SAMPLES 65536
using namespace Legion;

LegionRuntime::Logger::Category log_app("DLRM");

void register_custom_tasks()
{
  // dont need to register anything for this
}

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
    BMMTestMeta(int m, int k, int n, int d) {
        m = m, k = k, n = n, d = d;
    }
};

BMMTestMeta get_test_meta(const std::string file_path) {
    std::fstream myfile(file_path, std::ios_base::in);
    int m,k,n,d;
    myfile >> m >> k >> n >> d;
    BMMTestMeta meta(m,k,n,d);
    return meta;
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
  int m = 3;
  int k = 4;
  int n = 1;
  int d = 2;


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

    const int dims[] = {d,m,k};
    // sadly we have to pass batch_matmul 3-dimensional stretegy in this way for now to handle 3 dimensional tensor
    dense_input1 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  Tensor dense_input2;
  {
    const int dims[] = {d,n,k};
    // sadly we have to pass batch_matmul 3-dimensional stretegy in this way for now to handle 3 dimensional tensor
    dense_input2 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  // we can only use zero initializer because others don't support 3-dimensional tensor
  Initializer* initializer = new UniformInitializer(0, 0, 1);
  initializer->init(ffConfig.lg_ctx, runtime, &dense_input1);
  initializer->init(ffConfig.lg_ctx, runtime, &dense_input2);
  Tensor batch_matmul_ret = ff.batch_matmul("batch_matmul", dense_input1, dense_input2, true, false);

  ff.init_layers();
  // Data Loader

  // data_loader.next_random_batch(ff);
  ff.forward();
  // ff.zero_gradients(); // dont need to call this because there's no weights in batch_matmul
  ff.backward();
}


