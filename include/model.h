/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef _FLEXFLOW_RUNTIME_H_
#define _FLEXFLOW_RUNTIME_H_
#include "legion.h"
#include "config.h"
#include "initializer.h"
#include "optimizer.h"
#include "accessor.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <unistd.h>

using namespace Legion;

enum TaskIDs {
  /*
  ATTENTION: DO NOT ADD MORE TASK ENUMS HERE!!
  ADD NEW TASK ENUMS TO TaskIDs2!!
  TODO: figure out which task IDs are reserved,
  so far we know that we can't set TOP_LEVEL_TASK_ID to arbitrary integer
  */
  TOP_LEVEL_TASK_ID,
  FF_INIT_TASK_ID,
  IMAGE_INIT_TASK_ID,
  LABEL_INIT_TASK_ID,
  LOAD_IMAGES_TASK_ID,
  NORMALIZE_IMAGES_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_INIT_PARA_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  CONV2D_UPD_TASK_ID,
  EMBED_INIT_TASK_ID,
  EMBED_FWD_TASK_ID,
  EMBED_BWD_TASK_ID,
  POOL2D_INIT_TASK_ID,
  POOL2D_FWD_TASK_ID,
  POOL2D_BWD_TASK_ID,
  BATCHNORM_INIT_TASK_ID,
  BATCHNORM_INIT_PARA_TASK_ID,
  BATCHNORM_FWD_TASK_ID,
  BATCHNORM_BWD_TASK_ID,
  BATCHMATMUL_INIT_TASK_ID,
  BATCHMATMUL_FWD_TASK_ID,
  BATCHMATMUL_BWD_TASK_ID,
  TRANSPOSE_INIT_TASK_ID,
  TRANSPOSE_FWD_TASK_ID,
  TRANSPOSE_BWD_TASK_ID,
  LINEAR_INIT_TASK_ID,
  LINEAR_INIT_PARA_TASK_ID,
  LINEAR_FWD_TASK_ID,
  LINEAR_BWD_TASK_ID,
  LINEAR_BWD2_TASK_ID,
  LINEAR_UPD_TASK_ID,
  FLAT_INIT_TASK_ID,
  FLAT_FWD_TASK_ID,
  FLAT_BWD_TASK_ID,
  SOFTMAX_INIT_TASK_ID,
  SOFTMAX_FWD_TASK_ID,
  SOFTMAX_BWD_TASK_ID,
  CONCAT_INIT_TASK_ID,
  CONCAT_FWD_TASK_ID,
  CONCAT_BWD_TASK_ID,
  MSELOSS_BWD_TASK_ID,
  UPDATE_METRICS_TASK_ID,
  DUMMY_TASK_ID,
  // Optimizer
  SGD_UPD_TASK_ID,
  ADAM_UPD_TASK_ID,
  // Initializer
  GLOROT_INIT_TASK_ID,
  ZERO_INIT_TASK_ID,
  UNIFORM_INIT_TASK_ID,
  NORMAL_INIT_TASK_ID,
  // tensor helper tasks
  INIT_TENSOR_FROM_FILE_CPU_TASK,
  INIT_TENSOR_1D_FROM_FILE_CPU_TASK,
  INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
  INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
  INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
  DUMP_TENSOR_CPU_TASK,
  DUMP_TENSOR_1D_CPU_TASK,
  DUMP_TENSOR_2D_CPU_TASK,
  DUMP_TENSOR_3D_CPU_TASK,
  DUMP_TENSOR_4D_CPU_TASK,
  // Custom tasks
  CUSTOM_GPU_TASK_ID_FIRST,
  CUSTOM_GPU_TASK_ID_1,
  CUSTOM_GPU_TASK_ID_2,
  CUSTOM_GPU_TASK_ID_3,
  CUSTOM_GPU_TASK_ID_4,
  CUSTOM_GPU_TASK_ID_5,
  CUSTOM_GPU_TASK_ID_6,
  CUSTOM_GPU_TASK_ID_7,
  CUSTOM_GPU_TASK_ID_8,
  CUSTOM_GPU_TASK_ID_LAST,
  CUSTOM_CPU_TASK_ID_FIRST,
  CUSTOM_CPU_TASK_ID_1,
  CUSTOM_CPU_TASK_ID_2,
  CUSTOM_CPU_TASK_ID_3,
  CUSTOM_CPU_TASK_ID_4,
  CUSTOM_CPU_TASK_ID_5,
  CUSTOM_CPU_TASK_ID_6,
  CUSTOM_CPU_TASK_ID_7,
  CUSTOM_CPU_TASK_ID_LAST
};

enum ShardingID {
  DataParallelShardingID = 135,
};



enum ActiMode {
  AC_MODE_NONE,
  AC_MODE_RELU,
  AC_MODE_SIGMOID,
  AC_MODE_TANH,
};

enum AggrMode {
  AGGR_MODE_NONE,
  AGGR_MODE_SUM,
  AGGR_MODE_AVG,
};

enum PoolType {
  POOL_MAX,
  POOL_AVG,
};

enum DataType {
  DT_FLOAT,
  DT_DOUBLE,
  DT_INT32,
  DT_INT64,
  DT_BOOLEAN,
};

enum FieldIDs {
  FID_DATA,
};



enum TaskIDs2 {
  FIRST_TASK_ID = 99999,
  // FIRST_TASK_ID,
  RESHAPE_2_TO_3_INIT_TASK_ID,
  RESHAPE_2_TO_3_FWD_TASK_ID,
  RESHAPE_3_TO_2_FWD_TASK_ID,
  RESHAPE_3_TO_2_BWD_TASK_ID,
  RESHAPE_3_TO_2_INIT_TASK_ID,
  RESHAPE_2_TO_3_BWD_TASK_ID,
  TANH_1D_INIT_TASK_ID,
  TANH_2D_INIT_TASK_ID,
  TANH_3D_INIT_TASK_ID,
  TANH_1D_FWD_TASK_ID,
  TANH_2D_FWD_TASK_ID,
  TANH_3D_FWD_TASK_ID,
  TANH_1D_BWD_TASK_ID,
  TANH_2D_BWD_TASK_ID,
  TANH_3D_BWD_TASK_ID,
  First = FIRST_TASK_ID
  // Last = RESHAPE_2_TO_3_BWD_TASK_ID
};
struct PerfMetrics
{
  float train_loss;
  int train_all, train_correct, test_all, test_correct, val_all, val_correct;
};

struct FFHandler {
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  void *workSpace;
  size_t workSpaceSize;
};

struct Tensor {
  Tensor(void) {
    numDim = 0;
    for (int i = 0; i < MAX_DIM; i++) {
      adim[i] = 0;
      pdim[i] = 0;
    }
    region = LogicalRegion::NO_REGION;
    region_grad = LogicalRegion::NO_REGION;
    part = LogicalPartition::NO_PART;
    part_grad = LogicalPartition::NO_PART;
  }
  int numDim, adim[MAX_DIM], pdim[MAX_DIM];
  LogicalRegion region, region_grad;
  LogicalPartition part, part_grad;
};

class OpMeta {
public:
  OpMeta(FFHandler _handle) : handle(_handle) {};
public:
  FFHandler handle;
};

class FFModel;
class DataLoader;

class Op {
public:
  Op(const std::string& _name, const Tensor& input);
  Op(const std::string& _name, const Tensor& input1, const Tensor& input2);
  Op(const std::string& _name, int num, const Tensor* inputs);

  virtual void prefetch(const FFModel&);
  virtual void init(const FFModel&) = 0;
  virtual void forward(const FFModel&) = 0;
  virtual void backward(const FFModel&) = 0;
  //virtual void update(const FFModel&) = 0;
public:
  char name[MAX_OPNAME];
  IndexSpace task_is;
  Tensor output;
  Tensor inputs[MAX_NUM_INPUTS];
  bool trainableInputs[MAX_NUM_INPUTS];
  bool resetInputGrads[MAX_NUM_INPUTS];
  LogicalPartition input_lps[MAX_NUM_INPUTS], input_grad_lps[MAX_NUM_INPUTS];
  //Tensor locals[MAX_NUM_LOCALS];
  OpMeta* meta[MAX_NUM_WORKERS];
  int numLocals, numInputs;
};

class Parameter {
public:
  Tensor tensor;
  Op* op;
};

class FFModel {
public:
  FFModel(FFConfig &config);

  // Add a 2D convolutional layer
  Tensor conv2d(std::string name,
                const Tensor& input,
                int outChannels,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                ActiMode activation = AC_MODE_NONE,
                bool use_bias = true,
                Initializer* krenel_initializer = NULL,
                Initializer* bias_initializer = NULL);
  // Add an embedding layer
  Tensor embedding(const std::string& name,
                   const Tensor& input,
                   int num_entires, int outDim,
                   AggrMode aggr,
                   Initializer* kernel_initializer);
  // Add a 2D pooling layer
  Tensor pool2d(const std::string& name,
                const Tensor& input,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE);
  // Add a batch_norm layer
  Tensor batch_norm(std::string name,
                    Tensor input,
                    bool relu = true);
  // Add a dense layer
  Tensor dense(std::string name,
               const Tensor& input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               Initializer* kernel_initializer = NULL,
               Initializer* bias_initializer = NULL);
  // Add a linear layer
  Tensor linear(std::string name,
                const Tensor& input,
                int outDim,
                ActiMode activation = AC_MODE_NONE,
                bool use_bias = true,
                Initializer* kernel_initializer = NULL,
                Initializer* bias_initializer = NULL);

  // Add a batch matmul layer
  Tensor batch_matmul(std::string name,
                      const Tensor& input1,
                      const Tensor& input2,
                      const bool trans1=true,
                      const bool trans2=false);

  // Add a reshape layer
  template <int IDIM, int ODIM>
  Tensor reshape(std::string name,
                const Tensor& input,
                const int output_shape[]);

  // Add a concat layer
  Tensor concat(std::string name,
                int n, const Tensor* tensors,
                int axis);

  // Add a transpose layer
  Tensor transpose(std::string name, Tensor input);

  // Add a dot compressor layer
  Tensor dot_compressor(std::string name,
                        int num_dense_embeddings,
                        int num_sparse_embeddings, 
                        Tensor* _dense_embeddings, 
                        Tensor* _sparse_embeddings,
                        Tensor& dense_projection, 
                        int compressed_num_channels,
                        ActiMode activation = AC_MODE_NONE,
                        Initializer* kernel_initializer = NULL,
                        Initializer* bias_initializer = NULL,
                        bool use_bias = true,
                        Tensor* _kernel = NULL,
                        Tensor* _bias = NULL);

  // Add a flat layer
  Tensor flat(std::string name, Tensor input);

  // Add a softmax layer
  Tensor softmax(std::string name,
                 const Tensor& input,
                 const Tensor& label);

  // Add a tanh layer
  template<int DIM>
  Tensor tanh(std::string name, 
    const Tensor& input,
    const int output_shape[]);

  void mse_loss(const std::string& name,
                const Tensor& logits,
                const Tensor& labels,
                const std::string& reduction);

  void mse_loss3d(const std::string& name,
                const Tensor& logits,
                const Tensor& labels,
                const std::string& reduction);

  template<int NDIM>
  Tensor create_tensor(const int* dims,
                       const std::string& pc_name,
                       DataType data_type,
                       bool create_grad = true);

  template<int NDIM>
  void create_disjoint_partition(const Tensor& tensor,
                                 const IndexSpace& part_is,
                                 LogicalPartition& part_fwd,
                                 LogicalPartition& part_bwd);

  template<int NDIM, int TDIM>
  void create_data_parallel_partition_with_diff_dims(const Tensor& tensor,
                                                     const IndexSpaceT<TDIM>& task_is,
                                                     LogicalPartition& part_fwd,
                                                     LogicalPartition& part_bwd);
  template<int NDIM>
  Tensor create_tensor(const int* dims,
                       const IndexSpace& part_is,
                       DataType data_type,
                       bool create_grad = true);
  template<int NDIM>
  Tensor create_conv_weight(const int* dims,
                            const IndexSpaceT<4>& part_is,
                            DataType data_type,
                            Initializer* initializer,
                            bool create_grad = true);
  template<int NDIM>
  Tensor create_linear_weight(const int* dims,
                              const IndexSpaceT<2>& part_is,
                              DataType data_type,
                              Initializer* initializer,
                              bool create_grad = true);
  template<int NDIM>
  Tensor create_linear_replica(const int* dims,
                               const IndexSpaceT<2>& part_is,
                               DataType data_type);
  static PerfMetrics update_metrics_task(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, Runtime *runtime);
  void reset_metrics();
  void init_layers();
  void prefetch();
  void forward();
  void backward();
  void update();
  void zero_gradients();
  // Internal funcitons
  IndexSpace get_or_create_task_is(ParallelConfig pc);
  IndexSpace get_or_create_task_is(const Domain& domain);
  IndexSpace get_or_create_task_is(int ndims, const std::string& pcname);
  IndexSpace get_task_is(const Domain& domain) const;
public:
  FFConfig config;
  Optimizer* optimizer;
  //Tensor inputImage, inputRaw, inputLabel;
  std::vector<Op*> layers;
  std::vector<Parameter> parameters;
  FFHandler handlers[MAX_NUM_WORKERS];
  Future current_metrics;
  //DataLoader *dataLoader;
private:
  std::map<ParallelConfig, IndexSpace, ParaConfigCompare> taskIs;
};

class Conv2D : public Op {
public:
  Conv2D(FFModel& model, const std::string& pcname,
         const Tensor& input, int out_dim,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         ActiMode activation,
         bool use_bias,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  Tensor kernel, bias;
  bool profiling;
  ActiMode activation;
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler) : OpMeta(handler) {};
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
  bool relu, first_layer;
};

class Pool2D : public Op {
public:
  Pool2D(FFModel& model,
         const std::string& name,
         const Tensor& input,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, ActiMode _activation);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
  bool profiling;
};

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  bool relu;
};

class BatchNorm : public Op {
public:
  BatchNorm(std::string name, FFConfig config,
            Tensor input, IndexSpaceT<4> part_is,
            bool relu);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void init_para_task(const Task *task,
                             const std::vector<PhysicalRegion> &regions,
                             Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  bool relu, profiling;
  int num_replica;
  Tensor locals[MAX_NUM_LOCALS];
};

class BatchNormMeta : public OpMeta {
public:
  BatchNormMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t inputTensor, outputTensor, biasTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnBatchNormMode_t mode;
  float *runningMean, *runningVar, *saveMean, *saveVar;
  bool relu;
};

class Linear : public Op {
public:
  Linear(FFModel& model,
         const std::string& pcname,
         const Tensor& input,
         int outChannels,
         ActiMode activation,
         bool use_bias,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  //static void init_para_task(const Task *task,
  //                           const std::vector<PhysicalRegion> &regions,
  //                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  //static void update_task(const Task *task,
  //                        const std::vector<PhysicalRegion> &regions,
  //                        Context ctx, Runtime *runtime);
public:
  Tensor kernel, bias, replica;
  bool profiling;
  ActiMode activation;
};

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  const float *one_ptr;
};

class Embedding : public Op {
public:
  Embedding(FFModel& model,
            const std::string& pcname,
            const Tensor& input,
            int num_entries, int outDim,
            AggrMode _aggr,
            Initializer* kernel_initializer);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void forward_task_cpu(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void backward_task_cpu(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime);
public:
  Tensor kernel;
  AggrMode aggr;
  bool profiling;
};


class Flat : public Op {
public:
  Flat(FFModel& model,
       const std::string& pcname,
       const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
};

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle) {};
};

class Softmax : public Op {
public:
  Softmax(FFModel& model,
          const std::string& pcname,
          const Tensor& logit,
          const Tensor& label);

  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  bool profiling;
};

class SoftmaxMeta : public OpMeta {
public:
  SoftmaxMeta(FFHandler handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
#endif
};

class Concat : public Op {
public:
  Concat(FFModel& model,
         const std::string& name,
         int n, const Tensor* inputs, int axis);

  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  int axis;
  bool profiling;
};

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
};

class MSELoss : public Op {
public:
  MSELoss(FFModel& model,
          const std::string& pc_name,
          const Tensor& logit,
          const Tensor& label,
          AggrMode aggr);

  void init(const FFModel& model);
  void forward(const FFModel& model);
  void backward(const FFModel& model);
  //void update(const FFModel& model);

  static PerfMetrics backward_task(const Task *task,
                                   const std::vector<PhysicalRegion> &regions,
                                   Context ctx, Runtime *runtime);
public:
  AggrMode aggr_mode;
  bool profiling;
};

class UtilityTasks {
public:
  static FFHandler init_cuda_task(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx, Runtime *runtime);
  static void dummy_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime);
  static void init_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void init_labels_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void load_images_task(const Task *task,
                               const std::vector<PhysicalRegion> &regions,
                               Context ctx, Runtime *runtime);
  static void normalize_images_task(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
};

#ifdef DEADCODE
struct Sample {
  int label;
  char file[MAX_FILE_LENGTH];
};

struct DataLoadMeta {
  int numSamples;
  Sample samples[MAX_SAMPLES_PER_LOAD];
};

// class DataLoader
class DataLoader {
public:
  DataLoader(std::string);
  bool get_samples(int numSamples, DataLoadMeta &meta);
  bool shuffle_samples(void);
public:
  std::vector<Sample> samples;
  std::vector<Sample>::const_iterator sampleIter;
};
#endif

void top_level_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime);

void data_load_task(const Task* task,
                    const std::vector<PhysicalRegion>& regions,
                    Context ctx, Runtime* runtime);

void register_custom_tasks();

class BatchMatmul : public Op {
public:
  BatchMatmul(FFModel& model,
         const std::string& pcname,
         const Tensor& input1,
         const Tensor& input2,
         const bool trans1=true, // default matmul is C=A^T*B , where assume input layout are (d,k,m) , (d,k,n) and (d,m,n)
         const bool trans2=false);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(
                          const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime
                          );
public:
  IndexSpaceT<3> task_is;
  Tensor output, input1, input2;
  cublasOperation_t transpose_1, transpose_2;
  bool transpose_1_flag, transpose_2_flag;
  bool profiling;
};

class BatchMatmulMeta : public OpMeta {
public:
  BatchMatmulMeta(FFHandler handle) : OpMeta(handle) {};
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  const float *one_ptr;
};


class Transpose : public Op {
public:
  Transpose(FFModel& model,
         const std::string& pcname,
         const Tensor& _input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(
                          const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime
                          );
public:
  IndexSpaceT<3> task_is;
  Tensor output, input;
  bool profiling;
};

class TransposeMeta : public OpMeta {
public:
  TransposeMeta(FFHandler handle) : OpMeta(handle) {};
};

template <int IDIM, int ODIM>
class Reshape : public Op {
public:
  Reshape(FFModel& model,
         const std::string& pcname,
         const Tensor& _input,
         const int output_shape[]);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(
                          const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime
                          );
  static OpMeta* init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
public:
  Tensor input;
  bool profiling;
  std::string pcname;
  IndexSpaceT<ODIM> task_is;
};


class ReshapeMeta : public OpMeta {
public:
  ReshapeMeta(FFHandler handle) : OpMeta(handle) {};
};

template <int DIM>
class Tanh : public Op {
public:
  Tanh(FFModel& model,
         const std::string& pcname,
         const Tensor& _input, const int output_shape[]);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(
                          const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime
                          );
  static OpMeta* init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
public:
  Tensor input;
  bool profiling;
  std::string pcname;
  IndexSpaceT<DIM> task_is;
};

class TanhMeta : public OpMeta {
public:
  TanhMeta(FFHandler handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t activation;
#endif
};


#endif//_FLEXFLOW_RUNTIME_H_
