/* Copyright 2020 Stanford
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
#ifndef _FLEXFLOW_MODEL_H_
#define _FLEXFLOW_MODEL_H_
#include "legion.h"
#include "config.h"
#include "initializer.h"
#include "simulator.h"
#include "optimizer.h"
#include "accessor.h"
#include "loss_functions.h"
#include "metrics_functions.h"
#include <cuda_runtime.h>
#include <curand.h>
<<<<<<< HEAD
#include <cublas_v2.h>
#include <cuda_fp16.h>
=======
>>>>>>> 131466e75c28cc9c63006996f962ebdbf895fa9f
#include <unistd.h>

using namespace Legion;

#include "ffconst.h"

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
  ELEMENTBINARY_INIT_TASK_ID,
  ELEMENTBINARY_FWD_TASK_ID,
  ELEMENTBINARY_BWD_TASK_ID,
  ELEMENTUNARY_INIT_TASK_ID,
  ELEMENTUNARY_FWD_TASK_ID,
  ELEMENTUNARY_BWD_TASK_ID,
  CONV2D_INIT_TASK_ID,
  CONV2D_INIT_PARA_TASK_ID,
  CONV2D_FWD_TASK_ID,
  CONV2D_BWD_TASK_ID,
  CONV2D_UPD_TASK_ID,
  DROPOUT_INIT_TASK_ID,
  DROPOUT_FWD_TASK_ID,
  DROPOUT_BWD_TASK_ID,
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
  SPLIT_INIT_TASK_ID,
  SPLIT_FWD_TASK_ID,
  SPLIT_BWD_TASK_ID,
  RESHAPE_INIT_TASK_ID,
  RESHAPE_FWD_TASK_ID,
  RESHAPE_BWD_TASK_ID,
  REVERSE_INIT_TASK_ID,
  REVERSE_FWD_TASK_ID,
  REVERSE_BWD_TASK_ID,
  TRANSPOSE_INIT_TASK_ID,
  TRANSPOSE_FWD_TASK_ID,
  TRANSPOSE_BWD_TASK_ID,
  MSELOSS_BWD_TASK_ID,
  //Metrics tasks
  METRICS_COMP_TASK_ID,
  UPDATE_METRICS_TASK_ID,
  DUMMY_TASK_ID,
  // Loss
  LOSS_BWD_TASK_ID,
  // Optimizer
  SGD_UPD_TASK_ID,
  ADAM_UPD_TASK_ID,
  // Initializer
  GLOROT_INIT_TASK_ID,
  ZERO_INIT_TASK_ID,
  CONSTANT_INIT_TASK_ID,
  UNIFORM_INIT_TASK_ID,
  NORMAL_INIT_TASK_ID,
  // Search
  STRATEGY_SEARCH_TASK_ID,
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
  ACTIVATION_1D_INIT_TASK_ID,
  ACTIVATION_2D_INIT_TASK_ID,
  ACTIVATION_3D_INIT_TASK_ID,
  ACTIVATION_1D_FWD_TASK_ID,
  ACTIVATION_2D_FWD_TASK_ID,
  ACTIVATION_3D_FWD_TASK_ID,
  ACTIVATION_1D_BWD_TASK_ID,
  ACTIVATION_2D_BWD_TASK_ID,
  ACTIVATION_3D_BWD_TASK_ID,
  SIGMOID_CROSS_ENTROPY_WITH_LOGIT_BWD_TASK_ID,
  First = FIRST_TASK_ID
  // Last = RESHAPE_2_TO_3_BWD_TASK_ID
};
<<<<<<< HEAD
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
#ifndef DISABLE_COMPUTATION
  cudnnActivationMode_t mode;
#endif
};
=======
>>>>>>> 131466e75c28cc9c63006996f962ebdbf895fa9f

struct Tensor {
  Tensor(void) {
    numDim = 0;
    for (int i = 0; i < MAX_TENSOR_DIM; i++) {
      adim[i] = 0;
      //pdim[i] = 0;
    }
    region = LogicalRegion::NO_REGION;
    region_grad = LogicalRegion::NO_REGION;
    part = LogicalPartition::NO_PART;
    part_grad = LogicalPartition::NO_PART;
    owner_op = NULL;
    owner_idx = 0;
  }
  void inline_map(FFConfig &config);
  void inline_unmap(FFConfig &config);
  template<typename T>
  T* get_raw_ptr(FFConfig &config);
  void attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major);
  void detach_raw_ptr(FFConfig &config);
  bool get_input_sub_tensor(const ParallelConfig& pc,
                            Tensor& tensor,
                            OperatorType type);
  bool get_output_sub_tensor(const ParallelConfig& pc,
                             Tensor& tensor,
                             OperatorType type);
  size_t get_volume();
  int numDim, adim[MAX_TENSOR_DIM];
  DataType data_type;
  // Describes the ownership of this tensor
  Op* owner_op;
  int owner_idx;
  // The following fields are initialized after model.compile
  LogicalRegion region, region_grad;
  LogicalPartition part, part_grad;
  PhysicalRegion physical_region;
};

struct Parameter : Tensor {
  Parameter(void) {}
  template <typename T>
  bool set_weights(const FFModel& model,
                   const std::vector<int>& dims,
                   const T* data);
  template <typename T>
  bool get_weights(const FFModel& model,
                   T* data);
  std::vector<int> get_dims();
  std::string pcname; // indicating how the parameter is parallelized
  // Op* op; // Pointer to the operator that owns this parameter
};

class OpMeta {
public:
  OpMeta(FFHandler _handle) : handle(_handle) {};
public:
  FFHandler handle;
};

class Op {
public:
  Op(FFModel& model, OperatorType type, const std::string& _name, const Tensor& input);
  Op(FFModel& model, OperatorType type, const std::string& _name, const Tensor& input1, const Tensor& input2);
  Op(FFModel& model, OperatorType type, const std::string& _name, int num, const Tensor* inputs);
  Op(FFModel& model, OperatorType type, const std::string& _name, int num);

  Op(FFModel& model, OperatorType type, const Op* shared_op, const std::string& _name, const Tensor& input);
  // Pure virtual functions that must be implemented
  virtual void init(const FFModel&) = 0;
  virtual void forward(const FFModel&) = 0;
  virtual void backward(const FFModel&) = 0;
  virtual void create_weights(FFModel& model) = 0;
  virtual void create_output_and_partition(FFModel& model) = 0;
  virtual void print_layer(const FFModel& model) = 0;
  virtual bool measure_compute_time(Simulator* sim,
      const ParallelConfig& pc, float& forward, float& backward) = 0;
  virtual Tensor init_inout(FFModel&, const Tensor&) = 0;
  // Other virtual functions that can be optionally overwritten
  virtual ParallelConfig get_random_parallel_config(const FFModel& ff) const;
  virtual ParallelConfig get_data_parallel_config(const FFModel& ff) const;
  virtual Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx);
  virtual Domain get_output_tensor_shape(const ParallelConfig& pc, int output_idx, int part_idx);
  virtual Domain get_weight_tensor_shape(const ParallelConfig& pc, int weight_idx, int part_idx);
  // Helper functions
  void prefetch(const FFModel&);
  void zero_grad(const FFModel&);
  Parameter* get_parameter(int index);
public:
  OperatorType op_type;
  char name[MAX_OPNAME];
  IndexSpace task_is;
  Tensor outputs[MAX_NUM_OUTPUTS];
  Tensor inputs[MAX_NUM_INPUTS];
  Parameter weights[MAX_NUM_WEIGHTS];
  //bool trainableInputs[MAX_NUM_INPUTS];
  //bool resetInputGrads[MAX_NUM_INPUTS];
  LogicalPartition input_lps[MAX_NUM_INPUTS], input_grad_lps[MAX_NUM_INPUTS];
  //Tensor locals[MAX_NUM_LOCALS];
  OpMeta* meta[MAX_NUM_WORKERS];
  int numInputs, numWeights, numOutputs;
};

class ElementBinary;
class ElementUnary;
class Conv2D;
class Pool2D;
class Flat;
class Linear;
class Embedding;

class FFModel {
public:
  FFModel(FFConfig &config);
  // C++ APIs for constructing models
  // Add an exp layer
  Tensor exp(const Tensor& x);
  // Add an add layer
  Tensor add(const Tensor& x,
             const Tensor& y);
  // Add a subtract layer
  Tensor subtract(const Tensor& x,
                  const Tensor& y);
  // Add a multiply layer
  Tensor multiply(const Tensor& x,
                  const Tensor& y);
  // Add a divide layer
  Tensor divide(const Tensor& x,
                const Tensor& y);
  // Add an activation layer
  Tensor relu(const Tensor& x);
  Tensor sigmoid(const Tensor& x);
  Tensor tanh(const Tensor& x);
  Tensor elu(const Tensor& x);
  // Add a 2D convolutional layer
  Tensor conv2d(const Tensor& input,
                int outChannels,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                ActiMode activation = AC_MODE_NONE,
                bool use_bias = true,
                const Op* shared_op = NULL,
                Initializer* krenel_initializer = NULL,
                Initializer* bias_initializer = NULL);
  // Add a dropout layer
  Tensor dropout(const Tensor& input,
                 float rate,
                 unsigned long long seed = 0);
  // Add an embedding layer
  Tensor embedding(const Tensor& input,
                   int num_entires, int outDim,
                   AggrMode aggr,
                   const Op* shared_op = NULL,
                   Initializer* kernel_initializer = NULL);
  // Add a 2D pooling layer
  Tensor pool2d(const Tensor& input,
                int kernelH, int kernelW,
                int strideH, int strideW,
                int paddingH, int paddingW,
                PoolType type = POOL_MAX,
                ActiMode activation = AC_MODE_NONE);
  // Add a batch_norm layer
  Tensor batch_norm(const Tensor& input,
                    bool relu = true);
  // Add a batch_matmul layer
  Tensor batch_matmul(const Tensor& A,
                      const Tensor& B);
  // Add a dense layer
  Tensor dense(const Tensor& input,
               int outDim,
               ActiMode activation = AC_MODE_NONE,
               bool use_bias = true,
               const Op* shared_op = NULL,
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
  Tensor batch_matmul(const Tensor& input1,
                      const Tensor& input2,
                      const bool trans1=true,
                      const bool trans2=false);

  // Add a reshape layer
  template <int IDIM, int ODIM>
  Tensor reshape(const Tensor& input,
                 const int output_shape[]);

  // Add a concat layer
  Tensor concat(int n, const Tensor* tensors,
                int axis);

  // Add a split layer
  void split(const Tensor& input, Tensor* outputs,
             const std::vector<int>& split, int axis);
  // Add a flat layer
  Tensor flat(const Tensor& input);
  // Add a softmax layer
<<<<<<< HEAD
  Tensor softmax(std::string name,
                 const Tensor& input,
                 const Tensor& label);

  // Add a tanh layer
  template<int DIM>
  Tensor tanh(std::string name, 
    const Tensor& input,
    const int output_shape[]);

  // Add a sigmoid layer
  template<int DIM>
  Tensor sigmoid(std::string name, 
    const Tensor& input,
    const int output_shape[]);

  // Add a relu layer
  template<int DIM>
  Tensor relu(std::string name, 
    const Tensor& input,
    const int output_shape[]);

  // Add a elu layer
  template<int DIM>
  Tensor elu(std::string name, 
    const Tensor& input,
    const int output_shape[]);

  // Add a identity layer
  template<int DIM>
  Tensor identity(std::string name, 
    const Tensor& input,
    const int output_shape[]);


  // mse loss layer
  void mse_loss(const std::string& name,
                const Tensor& logits,
                const Tensor& labels,
                const std::string& reduction);

  void mse_loss3d(const std::string& name,
                const Tensor& logits,
                const Tensor& labels,
                const std::string& reduction);

  // sigmoid loss layer
  template<int NDIM>
  void sigmoid_cross_entropy_loss(const std::string& name,
                const Tensor& logits,
                const Tensor& labels);

  // sigmoid loss layer
  template<int NDIM>
  void cross_entropy_loss(const std::string& name,
                const Tensor& logits,
                const Tensor& labels);

=======
  Tensor softmax(const Tensor& input);
  // Create input tensors and constants
  Tensor transpose(const Tensor& input,
                   const std::vector<int>& perm);
  Tensor reshape(const Tensor& input,
                 const std::vector<int>& shape);
  Tensor reverse(const Tensor& input,
                 int axis);
>>>>>>> 131466e75c28cc9c63006996f962ebdbf895fa9f
  template<int NDIM>
  Tensor create_tensor(const int dims[],
                       DataType data_type,
                       const Op* owner_op = NULL,
                       bool create_grad = true);
  template<int NDIM>
  Tensor create_constant(const int dims[],
                         float value,
                         DataType date_type);
  // ========================================
  // Functional APIs for constructing models
  // ========================================
  ElementUnary* exp();
  ElementBinary* add();
  ElementBinary* subtract();
  ElementBinary* multiply();
  ElementBinary* divide();
  ElementUnary* relu();
  ElementUnary* sigmoid();
  ElementUnary* tanh();
  ElementUnary* elu();
  Conv2D* conv2d(int inChannels,
                 int outChannels,
                 int kernelH, int kernelW,
                 int strideH, int strideW,
                 int paddingH, int paddingW,
                 ActiMode activation = AC_MODE_NONE,
                 bool use_bias = true,
                 Initializer* krenel_initializer = NULL,
                 Initializer* bias_initializer = NULL);
  Embedding* embedding(int num_entires, int outDim,
                       AggrMode aggr,
                       Initializer* kernel_initializer);
  Pool2D* pool2d(int kernelH, int kernelW,
                 int strideH, int strideW,
                 int paddingH, int paddingW,
                 PoolType type = POOL_MAX,
                 ActiMode activation = AC_MODE_NONE);
  Linear* dense(int inDim, int outDim,
                ActiMode activation = AC_MODE_NONE,
                bool use_bias = true,
                Initializer* kernel_initializer = NULL,
                Initializer* bias_initializer = NULL);
  Flat* flat();
  // ========================================
  // Internal APIs that should not be invoked from applications
  // ========================================
  template<int NDIM>
  void create_disjoint_partition(const Tensor& tensor,
                                 const IndexSpaceT<NDIM>& part_is,
                                 LogicalPartition& part_fwd,
                                 LogicalPartition& part_bwd);

  template<int NDIM, int TDIM>
  void create_data_parallel_partition_with_diff_dims(const Tensor& tensor,
                                                     const IndexSpaceT<TDIM>& task_is,
                                                     LogicalPartition& part_fwd,
                                                     LogicalPartition& part_bwd);
  // Deprecated API --- to be removed
  //template<int NDIM>
  //Tensor create_tensor(const int* dims,
  //                     const IndexSpaceT<NDIM>& part_is,
  //                     DataType data_type,
  //                     bool create_grad = true);
  template<int NDIM>
  Parameter create_conv_weight(Op* op,
                               const int* dims,
                               const IndexSpaceT<4>& part_is,
                               DataType data_type,
                               Initializer* initializer,
                               bool create_grad = true);
  template<int NDIM, int TDIM>
  Parameter create_linear_weight(Op* op,
                                 const int* dims,
                                 const IndexSpaceT<TDIM>& part_is,
                                 DataType data_type,
                                 Initializer* initializer,
                                 bool create_grad = true);
  template<int NDIM, int TDIM>
  Tensor create_linear_replica(const int* dims,
                               const IndexSpaceT<TDIM>& part_is,
                               DataType data_type);
  static PerfMetrics update_metrics_task(const Task *task,
                                         const std::vector<PhysicalRegion> &regions,
                                         Context ctx, Runtime *runtime);
  void reset_metrics();
  void init_layers();
  void prefetch();
  void forward();
  void compute_metrics();
  void backward();
  void update();
  void compile(LossType loss_type, const std::vector<MetricsType>& metrics);
  void compile(Optimizer* optimizer, LossType loss_type, const std::vector<MetricsType>& metrics);
  void optimize(Simulator* simulator,
                std::map<Op*, ParallelConfig>& best,
                size_t budget, float alpha) const;
  void rewrite(const std::map<Op*, ParallelConfig>& current,
               std::map<Op*, ParallelConfig>& next) const;
  void zero_gradients();
  void print_layers(int id);
  // Internal funcitons
  Tensor get_tensor_from_guid(int guid);
  IndexSpace get_or_create_task_is(ParallelConfig pc);
  IndexSpace get_or_create_task_is(const Domain& domain);
  IndexSpace get_or_create_task_is(int ndims, const std::string& pcname);
  IndexSpace get_task_is(const Domain& domain) const;
public:
  int op_global_guid;
  FFConfig config;
  Optimizer* optimizer;
  Loss* loss_op;
  Metrics* metrics_op;
  Tensor label_tensor;
  //std::vector<Tensor> input_tensors;

  std::vector<Op*> layers;
  std::vector<Parameter> parameters;
  FFHandler handlers[MAX_NUM_WORKERS];
  Future current_metrics;
  //DataLoader *dataLoader;
private:
  bool debug;
  std::map<ParallelConfig, IndexSpace, ParaConfigCompare> taskIs;
};

class ElementBinaryMeta : public OpMeta {
public:
  ElementBinaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnOpTensorDescriptor_t opDesc;
  OperatorType op_type;
};

class ElementBinary : public Op {
public:
  ElementBinary(FFModel& model,
                OperatorType type,
                const Tensor& x,
                const Tensor& y);
  ElementBinary(FFModel& model,
                OperatorType type);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  void forward_kernel(const ElementBinaryMeta* m,
                      const float* in1_ptr,
                      const float* in2_ptr,
                      float* out_ptr) const;
  void backward_kernel(const ElementBinaryMeta* m,
                       const float* out_grad_ptr,
                       const float* in1_ptr,
                       const float* in2_ptr,
                       float* in1_grad_ptr,
                       float* in2_grad_ptr) const;
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  //IndexSpace task_is;
  OperatorType op_type;
};

class ElementUnaryMeta : public OpMeta {
public:
  ElementUnaryMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
};

class ElementUnary : public Op {
public:
  ElementUnary(FFModel& model,
               OperatorType type,
               const Tensor& x);
  ElementUnary(FFModel& model,
               OperatorType type);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
  bool use_cudnn() const;
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
};

class Conv2DMeta : public OpMeta {
public:
  Conv2DMeta(FFHandler handler);
  cudnnTensorDescriptor_t inputTensor, biasTensor, outputTensor;
  cudnnFilterDescriptor_t filterDesc;
  cudnnActivationDescriptor_t actiDesc;
  cudnnConvolutionDescriptor_t convDesc;
  cudnnConvolutionFwdAlgo_t fwdAlgo;
  cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
  cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
  bool relu;
};

class Conv2D : public Op {
public:
  Conv2D(FFModel& model,
         const Tensor& input,
         int out_dim,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         ActiMode activation,
         bool use_bias,
         const Op* shared_op,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Conv2D(FFModel& model,
         int in_dim, int out_dim,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         ActiMode activation,
         bool use_bias,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model);
  //Parameter* get_parameter(int index);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);


  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, HighLevelRuntime *runtime);
  void forward_kernel(const Conv2DMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr) const;
  void backward_kernel(const Conv2DMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr) const;
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  int in_channels, out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  bool profiling, use_bias;
  ActiMode activation;
  Initializer *kernel_initializer;
  Initializer *bias_initializer;
};

class DropoutMeta : public OpMeta {
public:
  DropoutMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnDropoutDescriptor_t dropoutDesc;
  void *reserveSpace, *dropoutStates;
  size_t reserveSpaceSize, dropoutStateSize;
};

class Dropout : public Op {
public:
  Dropout(FFModel& model,
          const Tensor& input,
          float rate,
          unsigned long long seed);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  //IndexSpaceT<4> task_is;
  float rate;
  unsigned long long seed;
  bool profiling;
};

class Pool2D : public Op {
public:
  Pool2D(FFModel& model,
         const Tensor& input,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, ActiMode _activation);
  Pool2D(FFModel& model,
         int kernelH, int kernelW,
         int strideH, int strideW,
         int paddingH, int paddingW,
         PoolType type, ActiMode _activation);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;
  bool profiling;
};

class Pool2DMeta : public OpMeta {
public:
  Pool2DMeta(FFHandler handle);
  cudnnTensorDescriptor_t inputTensor, outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  cudnnPoolingDescriptor_t poolDesc;
  bool relu;
};

class BatchNorm : public Op {
public:
  BatchNorm(FFModel& model, const Tensor& input, bool relu);

  Tensor init_inout(FFModel& model, const Tensor& input) { assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0);return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

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
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  bool relu, profiling;
  int num_replica;
  //Tensor locals[MAX_NUM_LOCALS];
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

class LinearMeta : public OpMeta {
public:
  LinearMeta(FFHandler handle, int batch_size);
  cudnnTensorDescriptor_t outputTensor;
  cudnnActivationDescriptor_t actiDesc;
  const float *one_ptr;
};

class Linear : public Op {
public:
  Linear(FFModel& model,
         const Tensor& input,
         int outChannels,
         ActiMode activation,
         bool use_bias,
         const Op* shared_op,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Linear(FFModel& model,
         int inChannels,
         int outChannels,
         ActiMode activation,
         bool use_bias,
         Initializer* kernel_initializer,
         Initializer* bias_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model);
  //Parameter* get_parameter(int index);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void backward2_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  void forward_kernel(const LinearMeta* m,
                      const float* input_ptr,
                      float* output_ptr,
                      const float* filter_ptr,
                      const float* bias_ptr,
                      int in_dim, int out_dim, int batch_size) const;
  void backward_kernel(const LinearMeta* m,
                       const float* input_ptr,
                       float* input_grad_ptr,
                       const float* output_ptr,
                       float* output_grad_ptr,
                       const float* kernel_ptr,
                       float* kernel_grad_ptr,
                       float* bias_ptr,
                       int in_dim, int out_dim, int batch_size) const;
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
  template<int NDIM>
  void create_weights_with_dim(FFModel& model);
  template<int NDIM>
  void init_with_dim(const FFModel& ff);
  template<int NDIM>
  void forward_with_dim(const FFModel& ff);
  template<int NDIM>
  void backward_with_dim(const FFModel& ff);
  template<int NDIM>
  static OpMeta* init_task_with_dim(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
  template<int NDIM>
  static void forward_task_with_dim(const Task *task,
                                    const std::vector<PhysicalRegion> &regions,
                                    Context ctx, Runtime *runtime);
  template<int NDIM>
  static void backward_task_with_dim(const Task *task,
                                     const std::vector<PhysicalRegion> &regions,
                                     Context ctx, Runtime *runtime);
  template<int NDIM>
  static void backward2_task_with_dim(const Task *task,
                                      const std::vector<PhysicalRegion> &regions,
                                      Context ctx, Runtime *runtime);
public:
  int in_channels, out_channels;
  Tensor replica;
  bool profiling, use_bias;
  ActiMode activation;
  Initializer *kernel_initializer;
  Initializer *bias_initializer;
};

class BatchMatmulMeta : public OpMeta {
public:
  BatchMatmulMeta(FFHandler handler);
};

class BatchMatmul : public Op {
public:
  BatchMatmul(FFModel& model,
              const Tensor& A,
              const Tensor& B);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  void forward_kernel(const BatchMatmulMeta* meta,
                      float* o_ptr,
                      const float* a_ptr,
                      const float* b_ptr,
                      const float* c_ptr,
                      int m, int n, int k, int batch) const;
  void backward_kernel(const BatchMatmulMeta* meta,
                       const float* o_ptr,
                       const float* o_grad_ptr,
                       const float* a_ptr,
                       float* a_grad_ptr,
                       const float* b_ptr,
                       float* b_grad_ptr,
                       float* c_grad_ptr,
                       int m, int n, int k, int batch) const;
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
  template<int NDIM>
  void init_with_dim(const FFModel& ff);
  template<int NDIM>
  void forward_with_dim(const FFModel& ff);
  template<int NDIM>
  void backward_with_dim(const FFModel& ff);
public:
  bool profiling;
};

class Embedding : public Op {
public:
  Embedding(FFModel& model,
            const Tensor& input,
            int num_entries, int outDim,
            AggrMode _aggr,
            const Op* shared_op,
            Initializer* kernel_initializer);
  Embedding(FFModel& model,
            int num_entries, int outDim,
            AggrMode _aggr,
            Initializer* kernel_initializer);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

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
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  //IndexSpaceT<2> task_is;
  int num_entries, out_channels;
  AggrMode aggr;
  bool profiling;
  Initializer* kernel_initializer;
};


class Flat : public Op {
public:
  Flat(FFModel& model,
       const Tensor& input);
  Flat(FFModel& model);
  Tensor init_inout(FFModel& model, const Tensor& input);
  //void add_to_model(FFModel& model);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);

  Domain get_input_tensor_shape(const ParallelConfig& pc, int input_idx, int part_idx);
public:
};

class FlatMeta : public OpMeta {
public:
  FlatMeta(FFHandler handle) : OpMeta(handle) {};
};

class Softmax : public Op {
public:
  Softmax(FFModel& model,
          const Tensor& logit);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
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

class Transpose : public Op {
public:
  Transpose(FFModel& model,
            const Tensor& input,
            const std::vector<int>& perm);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  int perm[MAX_TENSOR_DIM];
};

class Reverse : public Op {
public:
  Reverse(FFModel& model,
          const Tensor& input,
          int axis);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  int axis;
};

class Reshape : public Op {
public:
  Reshape(FFModel& model,
            const Tensor& input,
            const std::vector<int>& shape);
  Tensor init_inout(FFModel& model, const Tensor& input){assert(0); return Tensor();}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int IDIM, int ODIM>
  void create_output_and_partition_with_dim(FFModel& model);
};

class ConcatMeta : public OpMeta {
public:
  ConcatMeta(FFHandler handle) : OpMeta(handle) {};
};

class Concat : public Op {
public:
  Concat(FFModel& model,
         int n, const Tensor* inputs, int axis);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  int axis;
  bool profiling;
};

class Split : public Op {
public:
  Split(FFModel& model,
        const Tensor& input,
        const std::vector<int>& split,
        int axis);
  Tensor init_inout(FFModel& model, const Tensor& input) {assert(0); return Tensor();}
  //void add_to_model(FFModel& model) {assert(0);}
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  //void update(const FFModel&);
  void print_layer(const FFModel& model) {assert(0);}
  //Parameter* get_parameter(int index) {assert(0); return NULL;}
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
private:
  template<int NDIM>
  void create_output_and_partition_with_dim(FFModel& model);
public:
  int axis;
  //IndexSpace task_is;
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
              const Tensor& input1,
              const Tensor& input2,
              const bool trans1=true, // default matmul is C=A^T*B , where assume input layout are (d,k,m) , (d,k,n) and (d,m,n)
              const bool trans2=false);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
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
            const Tensor& _input);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
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
          const Tensor& _input,
          const int output_shape[]);
  Tensor init_inout(FFModel& model, const Tensor& input);
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  void print_layer(const FFModel& model);
  void create_weights(FFModel& model);
  void create_output_and_partition(FFModel& model);

  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static OpMeta* init_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  bool measure_compute_time(Simulator* sim,
                            const ParallelConfig& pc,
                            float& forward_time,
                            float& backward_time);
public:
  bool profiling;
};


class ReshapeMeta : public OpMeta {
public:
  ReshapeMeta(FFHandler handle) : OpMeta(handle) {};
};

<<<<<<< HEAD
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

template <int DIM>
class Loss : public Op {
public:
  Loss(
    FFModel& model,
    const std::string& pc_name, 
    const Tensor& logits,
    const Tensor& labels,
    const std::string& loss
  );
  void init(const FFModel&);
  void forward(const FFModel&);
  void backward(const FFModel&);
  static PerfMetrics backward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
public:
  Tensor input;
  bool profiling;
  std::string pcname;
  IndexSpaceT<DIM> task_is;
  std::string loss;
};

class SigmoidBinaryCrossEntropyWithLogicMeta : public OpMeta {
public:
  SigmoidBinaryCrossEntropyWithLogicMeta(FFHandler handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t activation;
  // cudnnActivationMode_t mode;
#endif
};

template <int DIM>
class Activation : public Op {
public:
  Activation(FFModel& model,
         const std::string& pcname,
         const std::string& _mode,
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
  std::string mode;
};

class ActivationMeta : public OpMeta {
public:
  ActivationMeta(FFHandler handle) : OpMeta(handle) {};
#ifndef DISABLE_COMPUTATION
  cudnnTensorDescriptor_t inputTensor;
  cudnnActivationDescriptor_t activation;
  // cudnnActivationMode_t mode;
#endif
};

#endif//_FLEXFLOW_RUNTIME_H_
=======
void register_c_custom_tasks();
#endif//_FLEXFLOW_MODEL_H_
>>>>>>> 131466e75c28cc9c63006996f962ebdbf895fa9f
