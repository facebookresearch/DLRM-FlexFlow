#include "batch_matmul_test.h"
#include "hdf5.h"
#include <sstream>
#include <fstream>
#include <iostream>


using namespace Legion;

LegionRuntime::Logger::Category log_app("DLRM");

void parse_input_args(char **argv, int argc, DLRMConfig& apConfig);

Tensor create_mlp(FFModel* model, const Tensor& input,
                  std::vector<int> ln, int sigmoid_layer)
{
  Tensor t = input;
  for (int i = 0; i < (int)(ln.size()-1); i++) {
    float std_dev = sqrt(2.0f / (ln[i+1] + ln[i]));
    Initializer* weight_init = new NormInitializer(std::rand(), 0, std_dev);
    std_dev = sqrt(2.0f / ln[i+1]);
    Initializer* bias_init = new NormInitializer(std::rand(), 0, std_dev);
    ActiMode activation = i == sigmoid_layer ? AC_MODE_SIGMOID : AC_MODE_RELU;
    t = model->dense("linear", t, ln[i+1], activation, true/*bias*/, weight_init, bias_init);
  }
  return t;
}

Tensor create_emb(FFModel* model, const Tensor& input,
                  int input_dim, int output_dim, int idx)
{
  float range = sqrt(1.0f / input_dim);
  Initializer* embed_init = new UniformInitializer(std::rand(), -range, range);
  return model->embedding("embedding"+std::to_string(idx), input, input_dim, output_dim, AGGR_MODE_SUM, embed_init);
}

Tensor interact_features(FFModel* model, const Tensor& x,
                         const std::vector<Tensor>& ly,
                         std::string interaction)
{
  // Currently only support cat
  // TODO: implement dot attention
  if (interaction == "cat") {
    Tensor* inputs = (Tensor*) malloc(sizeof(Tensor) * (1 + ly.size()));
    inputs[0] = x;
    for (size_t i = 0; i < ly.size(); i++)
      inputs[i+1] = ly[i];
    return model->concat("concat", ly.size() + 1, inputs, 1/*axis*/);
    free(inputs);
  } else {
    assert(false);
  }
}

void print_vector(const std::string& name, const std::vector<int>& vector)
{
  std::ostringstream out;
  for (size_t i = 0; i < vector.size() - 1; i++)
    out << vector[i] << " ";
  if (vector.size() > 0)
    out << vector[vector.size() - 1];
  log_app.print("%s: %s", name.c_str(), out.str().c_str());
}



void parse_input_args(char **argv, int argc, DLRMConfig& config)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--arch-sparse-feature-size")) {
      config.sparse_feature_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-embedding-size")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.embedding_size.clear();
      while (std::getline(ss, word, '-')) {
        config.embedding_size.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--embedding-bag-size")) {
      config.embedding_bag_size = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-bot")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.mlp_bot.clear();
      while (std::getline(ss, word, '-')) {
        config.mlp_bot.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--arch-mlp-top")) {
      std::stringstream ss(std::string(argv[++i]));
      std::string word;
      config.mlp_top.clear();
      while (std::getline(ss, word, '-')) {
        config.mlp_top.push_back(std::stoi(word));
      }
      continue;
    }
    if (!strcmp(argv[i], "--loss-threshold")) {
      config.loss_threshold = atof(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-top")) {
      config.sigmoid_top = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--sigmoid-bot")) {
      config.sigmoid_bot = atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--arch-interaction-op")) {
      config.arch_interaction_op = std::string(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--dataset")) {
      config.dataset_path = std::string(argv[++i]);
      continue;
    }
  }
}

DataLoader::DataLoader(FFModel& ff, const DLRMConfig& dlrm){
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  const int m = 32,n = 16,k = 128,d = 8;

    // for batch_matmul testing
  {
    const int dims[] = {d, m, n};
    batch_matmul_output = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  {
    const int dims[] = {d, m, k};
    batch_matmul_input1 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  {
    const int dims[] = {d, k, n};
    batch_matmul_input2 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }

  // we can only use zero initializer because others don't support 3-dimensional tensor
  Initializer* initializer = new ZeroInitializer();
  initializer->init(ctx, runtime, &batch_matmul_input1);
  initializer->init(ctx, runtime, &batch_matmul_input2);
  initializer->init(ctx, runtime, &batch_matmul_output);
    // CUSTOM_GPU_TASK_ID_8 is random_3d_batch
  const int num_dim = 3;
  std::string pc_name = "batch_matmul";
  IndexSpaceT<num_dim> task_is = IndexSpaceT<3>(ff.get_or_create_task_is(pc_name));
  Rect<num_dim> rect = runtime->get_index_space_domain(ctx, task_is);
  ArgumentMap argmap;
  int idx = next_index;
  for (PointInRectIterator<num_dim> it(rect); it(); it++) {
    SampleIdxs meta;
    // assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
    meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
    for (int i = 0; i < meta.num_samples; i++)
      meta.idxs[i] = idx++;
    argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
  }
  // CUSTOM_GPU_TASK_ID_8 is random_3d_batch
  IndexLauncher launcher(CUSTOM_GPU_TASK_ID_8, task_is,
                        TaskArgument(NULL, 0), argmap,
                        Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                        FFConfig::get_hash_id(std::string(pc_name)));

  launcher.add_region_requirement(
      RegionRequirement(batch_matmul_output.part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, batch_matmul_output.region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(batch_matmul_input1.part, 0/*projection id*/,
    READ_ONLY, EXCLUSIVE, batch_matmul_input1.region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(batch_matmul_input2.part, 0/*projection id*/,
    READ_ONLY, EXCLUSIVE, batch_matmul_input2.region));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


DataLoader::DataLoader(FFModel& ff,
                       const DLRMConfig& dlrm,
                       const std::vector<Tensor>& _sparse_inputs,
                       Tensor _dense_input, Tensor _label)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  num_samples = 0;
  if (dlrm.dataset_path == "") {
    log_app.print("Use random dataset...");
    num_samples = 256 * 4 * ff.config.workersPerNode * ff.config.numNodes;
    //num_samples = 256 * 2 * 8 * 16;
    log_app.print("Number of random samples = %d\n", num_samples);
  } else {
    log_app.print("Start loading dataset from %s", dlrm.dataset_path.c_str());
    hid_t file_id = H5Fopen(dlrm.dataset_path.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    // X_int
    {
      hsize_t dims[2], maxdims[2];
      hid_t x_int_dataset_id = H5Dopen2(file_id, "X_int", H5P_DEFAULT);
      hid_t x_int_space_id = H5Dget_space(x_int_dataset_id);
      hid_t x_int_type_id = H5Dget_type(x_int_dataset_id);
      assert(H5Sget_simple_extent_dims(x_int_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_int_type_id) == H5T_FLOAT);
      num_samples = dims[0];
      assert(dlrm.mlp_bot[0] == (int)dims[1]);
      H5Tclose(x_int_type_id);
      H5Dclose(x_int_dataset_id);
      H5Sclose(x_int_space_id);
    }
    // X_cat
    {
      hsize_t dims[2], maxdims[2];
      hid_t x_cat_dataset_id = H5Dopen2(file_id, "X_cat", H5P_DEFAULT);
      hid_t x_cat_space_id = H5Dget_space(x_cat_dataset_id);
      hid_t x_cat_type_id = H5Dget_type(x_cat_dataset_id);
      assert(H5Sget_simple_extent_dims(x_cat_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_cat_type_id) == H5T_INTEGER);
      assert(num_samples == (int)dims[0]);
      assert(_sparse_inputs.size() == dims[1]);
      H5Tclose(x_cat_type_id);
      H5Dclose(x_cat_dataset_id);
      H5Sclose(x_cat_space_id);
    }
    // y
    {
      hsize_t dims[2], maxdims[2];
      hid_t y_dataset_id = H5Dopen2(file_id, "y", H5P_DEFAULT);
      hid_t y_space_id = H5Dget_space(y_dataset_id);
      hid_t y_type_id = H5Dget_type(y_dataset_id);
      H5Sget_simple_extent_dims(y_space_id, dims, maxdims);
      assert(num_samples == (int)dims[0]);
      //assert(dims[1] == 1);
      H5Tclose(y_type_id);
      H5Dclose(y_dataset_id);
      H5Sclose(y_space_id);
    }
    H5Fclose(file_id);
    log_app.print("Finish loading dataset from %s", dlrm.dataset_path.c_str());
    log_app.print("Loaded %d samples", num_samples);
  }
  for (size_t i = 0; i < _sparse_inputs.size(); i++) {
    batch_sparse_inputs.push_back(_sparse_inputs[i]);
  }
  {
    const int dims[] = {num_samples, (int)_sparse_inputs.size()*dlrm.embedding_bag_size};
    full_sparse_input = ff.create_tensor<2>(dims, "", DT_INT64);
  }
  {
    batch_dense_input = _dense_input;
    const int dims[] = {num_samples, dlrm.mlp_bot[0]};
    full_dense_input = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  {
    batch_label = _label;
    const int dims[] = {num_samples, 1};
    full_label = ff.create_tensor<2>(dims, "", DT_FLOAT);
  }
  // Load entire dataset
  // TODO: Use index launcher instead of task launcher
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1,
      TaskArgument(dlrm.dataset_path.c_str(), dlrm.dataset_path.length()+10));
  // regions[0]: full_sparse_input
  launcher.add_region_requirement(
      RegionRequirement(full_sparse_input.region,
                        WRITE_ONLY, EXCLUSIVE, full_sparse_input.region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_dense_input
  launcher.add_region_requirement(
      RegionRequirement(full_dense_input.region,
                        WRITE_ONLY, EXCLUSIVE, full_dense_input.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[3]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label.region,
                        WRITE_ONLY, EXCLUSIVE, full_label.region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx,
                                     Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  // Note that these instances are in ZCM, can only use
  // TensorAccessorW with readOutput flag
  const AccessorWO<int64_t, 2> acc_sparse_input(regions[0], FID_DATA);
  const AccessorWO<float, 2> acc_dense_input(regions[1], FID_DATA);
  const AccessorWO<float, 2> acc_label_input(regions[2], FID_DATA);
  Rect<2> rect_sparse_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rect_dense_input = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rect_label_input = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_sparse_input.accessor.is_dense_arbitrary(rect_sparse_input));
  assert(acc_dense_input.accessor.is_dense_arbitrary(rect_dense_input));
  assert(acc_label_input.accessor.is_dense_arbitrary(rect_label_input));
  int64_t* sparse_input_ptr = acc_sparse_input.ptr(rect_sparse_input.lo);
  float* dense_input_ptr = acc_dense_input.ptr(rect_dense_input.lo);
  float* label_input_ptr = acc_label_input.ptr(rect_label_input.lo);
  int num_samples = rect_sparse_input.hi[1] - rect_sparse_input.lo[1] + 1;
  int num_sparse_inputs = rect_sparse_input.hi[0] - rect_sparse_input.lo[0] + 1;
  assert(num_samples == rect_dense_input.hi[1] - rect_dense_input.lo[1] + 1);
  int num_dense_dims = rect_dense_input.hi[0] - rect_dense_input.lo[0] + 1;
  assert(num_samples == rect_label_input.hi[1] - rect_label_input.lo[1] + 1);
  assert(rect_label_input.hi[0] == rect_label_input.lo[0]);
  std::string file_name((const char*)task->args);
  if (file_name.length() == 0) {
    log_app.print("Start generating random input samples");
    for (size_t i = 0; i < rect_sparse_input.volume(); i++)
      sparse_input_ptr[i] = std::rand() % 1000000;
    for (size_t i = 0; i < rect_dense_input.volume(); i++)
      dense_input_ptr[i] = ((float)std::rand()) / RAND_MAX;
    for (size_t i = 0; i < rect_label_input.volume(); i++)
      label_input_ptr[i] = std::rand() % 2;
  } else {
    hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    // Load X_cat
    {
      log_app.print("Start loading sparse features from "
                    "%s.%s", file_name.c_str(), "X_cat");
      hsize_t dims[2], maxdims[2];
      hid_t x_cat_dataset_id = H5Dopen2(file_id, "X_cat", H5P_DEFAULT);
      hid_t x_cat_space_id = H5Dget_space(x_cat_dataset_id);
      hid_t x_cat_type_id = H5Dget_type(x_cat_dataset_id);
      assert(H5Sget_simple_extent_dims(x_cat_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_cat_type_id) == H5T_INTEGER);
      assert(num_samples == (int)dims[0]);
      assert(num_sparse_inputs == (int)dims[1]);
      H5Dread(x_cat_dataset_id, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              sparse_input_ptr);
      H5Tclose(x_cat_type_id);
      H5Dclose(x_cat_dataset_id);
      H5Sclose(x_cat_space_id);
      log_app.print("Finish loading sparse features");
    }
    // Load X_int
    {
      log_app.print("Start loading dense features from "
                    "%s.%s", file_name.c_str(), "X_int");
      hsize_t dims[2], maxdims[2];
      hid_t x_int_dataset_id = H5Dopen2(file_id, "X_int", H5P_DEFAULT);
      hid_t x_int_space_id = H5Dget_space(x_int_dataset_id);
      hid_t x_int_type_id = H5Dget_type(x_int_dataset_id);
      assert(H5Sget_simple_extent_dims(x_int_space_id, dims, maxdims) == 2);
      assert(H5Tget_class(x_int_type_id) == H5T_FLOAT);
      num_samples = dims[0];
      assert(num_dense_dims == (int)dims[1]);
      H5Dread(x_int_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              dense_input_ptr);
      H5Tclose(x_int_type_id);
      H5Dclose(x_int_dataset_id);
      H5Sclose(x_int_space_id);
      log_app.print("Finish loading dense features");
    }
    // Load y
    {
      log_app.print("Start loading labels from "
                    "%s.%s", file_name.c_str(), "y");
      hsize_t dims[2], maxdims[2];
      hid_t y_dataset_id = H5Dopen2(file_id, "y", H5P_DEFAULT);
      hid_t y_space_id = H5Dget_space(y_dataset_id);
      hid_t y_type_id = H5Dget_type(y_dataset_id);
      H5Sget_simple_extent_dims(y_space_id, dims, maxdims);
      assert(num_samples == (int)dims[0]);
      //assert(dims[1] == 1);
      H5Dread(y_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
              label_input_ptr);
      H5Tclose(y_type_id);
      H5Dclose(y_dataset_id);
      H5Sclose(y_space_id);
      log_app.print("Finish loading labels");
    }
  }
}

void DataLoader::next_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  // Load Sparse Inputs
  for (size_t i = 0; i < batch_sparse_inputs.size(); i++) {
    int hash = batch_sparse_inputs.size() * 1000 + i;
    std::string pc_name = "embedding"+std::to_string(i);
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(pc_name));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      // Assert that we have enough slots to save the indices
      assert(meta.num_samples <= MAX_NUM_SAMPLES);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1, task_is,
                           TaskArgument(&hash, sizeof(int)), argmap,
                           Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                           FFConfig::get_hash_id(pc_name));
//#if 1
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_sparse_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_sparse_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
//#endif
    launcher.add_region_requirement(
        RegionRequirement(batch_sparse_inputs[i].part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_sparse_inputs[i].region));
    launcher.add_field(1, FID_DATA);
    //std::cout << "CUSTOM_CPU_TASK_ID_2" << std::endl;
    runtime->execute_index_space(ctx, launcher);
    //std::cout << "Done CUSTOM_CPU_TASK_ID_2" << std::endl;
  }
  // Load Dense Input
  {
    std::string pc_name = "";
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(pc_name));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(pc_name)));
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_dense_input.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_dense_input.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_dense_input.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_dense_input.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Labels
  {
    std::string pc_name = "";
    IndexSpaceT<2> task_is = IndexSpaceT<2>(ff.get_or_create_task_is(pc_name));
    Rect<2> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<2> it(rect); it(); it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_3, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(pc_name)));
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_label.region, 0/*projection id*/,
                          READ_ONLY, EXCLUSIVE, full_label.region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_label.region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // progress next_index
  next_index += ff.config.batchSize;
}

void DataLoader::shuffle()
{}

void DataLoader::reset()
{
  next_index = 0;
}
void DataLoader::load_sparse_input_cpu(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx,
                                   Runtime* runtime)
{
  std::cout << "load_sparse_input_cpu" << std::endl;
}

void register_custom_tasks()
{
  // Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load Sparse Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Sparse Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_sparse_input>(
        registrar, "Load Sparse Inputs Task");
  }
  // Load Dense Inputs
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2, "Load Dense Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_dense_input>(
        registrar, "Load Dense Inputs Task");
  }
  // Load Labels
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_3, "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Labels");
  }
  // Load batched input (random)
    {
      TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_8, "Load Batched Matrices");
      registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
      registrar.set_leaf();
      Runtime::preregister_task_variant<DataLoader::random_3d_batch>(
          registrar, "Load Batched Matrices");
    }
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
  DLRMConfig dlrmConfig;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    ffConfig.parse_args(argv, argc);
    // parse_input_args(argv, argc, dlrmConfig);
    // log_app.print("batchSize(%d) workersPerNodes(%d) numNodes(%d)",
    //     ffConfig.batchSize, ffConfig.workersPerNode, ffConfig.numNodes);
    // log_app.print("EmbeddingBagSize(%d)", dlrmConfig.embedding_bag_size);
    // print_vector("Embedding Vocab Sizes", dlrmConfig.embedding_size);
    // print_vector("MLP Top", dlrmConfig.mlp_top);
    // print_vector("MLP Bot", dlrmConfig.mlp_bot);
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
  DataLoader data_loader(ff, dlrmConfig);

  data_loader.reset();
  // data_loader.next_random_batch(ff);
  ff.forward();
  // ff.zero_gradients(); // dont need to call this because there's no weights in batch_matmul
  ff.backward();
}



void DataLoader::next_random_batch(FFModel& ff)
{
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;

  // Load Dense Input
  {
    const int num_dim = 3;
    std::string pc_name = "batch_matmul";
    IndexSpaceT<num_dim> task_is = IndexSpaceT<3>(ff.get_or_create_task_is(pc_name));
    Rect<num_dim> rect = runtime->get_index_space_domain(ctx, task_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (PointInRectIterator<num_dim> it(rect); it(); it++) {
      SampleIdxs meta;
      // assert(ff.config.batchSize % (rect.hi[1] - rect.lo[1] + 1) == 0);
      meta.num_samples = ff.config.batchSize / (rect.hi[1] - rect.lo[1] + 1);
      for (int i = 0; i < meta.num_samples; i++)
        meta.idxs[i] = idx++;
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    // CUSTOM_GPU_TASK_ID_8 is random_3d_batch
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_8, task_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(pc_name)));

    launcher.add_region_requirement(
        RegionRequirement(batch_matmul_output.part, 0/*projection id*/,
                          WRITE_ONLY, EXCLUSIVE, batch_matmul_output.region));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(batch_matmul_input1.part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, batch_matmul_input1.region));
    launcher.add_field(1, FID_DATA);
    launcher.add_region_requirement(
      RegionRequirement(batch_matmul_input2.part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, batch_matmul_input2.region));
    launcher.add_field(2, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }

  // progress next_index
  next_index += ff.config.batchSize;
}