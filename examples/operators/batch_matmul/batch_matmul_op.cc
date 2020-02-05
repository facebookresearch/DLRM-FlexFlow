#include "batch_matmul_op.h"

Tensor batch_matmul(FFModel* model, const Tensor& a,
                         const Tensor& b,)
{


    return model->batch_matmul("batch_matmul", a, b)
}



void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime)
{
  FFConfig ffConfig;
  // Parse input arguments
  DLRMConfig dlrmConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    parse_input_args(argv, argc, dlrmConfig);
    log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
        ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
    log_app.print("EmbeddingBagSize(%d)", dlrmConfig.embedding_bag_size);
    print_vector("Embedding Vocab Sizes", dlrmConfig.embedding_size);
    print_vector("MLP Top", dlrmConfig.mlp_top);
    print_vector("MLP Bot", dlrmConfig.mlp_bot);
  }

  ffConfig.lg_ctx = ctx;
  ffConfig.lg_hlr = runtime;
  ffConfig.field_space = runtime->create_field_space(ctx);
  FFModel ff(ffConfig);

  
  Tensor dense_input1;
  {
    const int dims[] = {ffConfig.batchSize, dlrmConfig.mlp_bot[0]};
    dense_input1 = ff.create_tensor<3>(dims, "", DT_FLOAT);
  }
  Tensor dense_input2;
  {
    const int dims[] = {ffConfig.batchSize, dlrmConfig.mlp_bot[0]};
    dense_input2 = ff.create_tensor<3>(dims, "", DT_FLOAT);
  }

    Tensor batch_matmul_ret = batch_matmul(&ff, dense_input1, dense_input2);
  

  ff.mse_loss("mse_loss"/*name*/, p, label, "average"/*reduction*/);
  // Use SGD Optimizer
  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  ff.init_layers();
  // Data Loader
  DataLoader data_loader(ff, dlrmConfig, sparse_inputs, dense_input, label);

  // Warmup iterations
  for (int iter = 0; iter < 1; iter++) {
    data_loader.reset();
    data_loader.random_3d_batch(ff);
    ff.forward();
  }

  
  {
    runtime->issue_execution_fence(ctx);
    TimingLauncher timer(MEASURE_MICRO_SECONDS);
    Future future = runtime->issue_timing_measurement(ctx, timer);
    future.get_void_result();
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  double run_time = 1e-6 * (ts_end - ts_start);
  printf("ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n", run_time,
         data_loader.num_samples * ffConfig.epochs / run_time);
}
