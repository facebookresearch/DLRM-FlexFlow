#include "model.h"
#include <algorithm>


Tensor FFModel::dot_compressor(
  std::string name,
  int num_dense_embeddings,
  int num_sparse_embeddings,
  Tensor* _dense_embeddings,
  Tensor* _sparse_embeddings, 
  Tensor& dense_projection,
  int compressed_num_channels,
  ActiMode activation,
  Initializer* kernel_initializer,
  Initializer* bias_initializer,
  bool use_bias
  ){
  assert(_dense_embeddings[0].numDim == 2);
  assert(num_dense_embeddings > 0);
  assert(_sparse_embeddings[0].numDim == 2);
  assert(num_sparse_embeddings > 0);
  int sparse_in_dim = _sparse_embeddings[0].adim[_sparse_embeddings[0].numDim-2];
  int dense_in_dim = _dense_embeddings[0].adim[_dense_embeddings[0].numDim-2];
  assert(sparse_in_dim == dense_in_dim);
  int num_channels = num_sparse_embeddings + num_dense_embeddings;
  int batch_size = _sparse_embeddings[0].adim[_sparse_embeddings[0].numDim-1];
  
  // merge two embedding lists
  std::vector<Tensor> dense_embeddings(_dense_embeddings,
   _dense_embeddings + num_dense_embeddings);
  std::vector<Tensor> sparse_embeddings(_sparse_embeddings,
   _sparse_embeddings + num_sparse_embeddings);
  std::vector<Tensor> embeddings;
  embeddings.insert(embeddings.begin(), sparse_embeddings.begin(), sparse_embeddings.end());
  embeddings.insert(embeddings.end(), dense_embeddings.begin(), dense_embeddings.end());

  // concat embeddings
  Concat *cat_input = new Concat(*this, "concat_input", num_channels, &embeddings[0], 1);
  layers.push_back(cat_input);
  
  // reshape 2 to 3
  int l1_shape[3] = {batch_size, num_channels, sparse_in_dim};
  Reshape<2, 3> *reshape_cat_input = new Reshape<2,3>(*this, 
    "reshape_concat_input", 
    cat_input->output, 
    l1_shape);
  layers.push_back(reshape_cat_input); 
  
  // transpose 
  Transpose *transpose_reshape_cat_input = new Transpose(*this, 
    "transpose_reshape_cat_input", 
    reshape_cat_input->output);
  layers.push_back(transpose_reshape_cat_input);

  // reshape 3 to 2
  int l2_shape[2] = {batch_size * sparse_in_dim, num_channels};
  Reshape<3, 2> *reshape_transpose_reshape_cat_input = new Reshape<3,2>(*this, 
    "reshape_transpose_reshape_cat_input", 
    transpose_reshape_cat_input->output,
    l2_shape);
  layers.push_back(reshape_transpose_reshape_cat_input);

  // linear layer
  if (kernel_initializer == NULL) {
    int seed = std::rand();
    kernel_initializer = new GlorotUniform(seed);
  }
  if (bias_initializer == NULL) {
    bias_initializer = new ZeroInitializer();
  }
  Linear *compressed_channels = new Linear(*this, "compressed channels", 
    reshape_transpose_reshape_cat_input->output, 
    compressed_num_channels, activation, use_bias,
    kernel_initializer, bias_initializer);
  layers.push_back(compressed_channels);
  Parameter kernel, bias;
  kernel.tensor = compressed_channels->kernel;
  kernel.op = compressed_channels;
  parameters.push_back(kernel);
  if (use_bias) {
    bias.tensor = compressed_channels->bias;
    bias.op = compressed_channels;
    parameters.push_back(bias);
  }

  // unpack channels - reshape 2 to 3
  int l3_shape[3] = {batch_size, sparse_in_dim, compressed_num_channels};
  Reshape<2, 3> *unpacked_compressed_channels = new Reshape<2,3>(*this, 
    "unpacked_compressed_channels", 
    compressed_channels->output, 
    l3_shape);
  layers.push_back(unpacked_compressed_channels); 

  // bmm 
  BatchMatmul *bmm = new BatchMatmul(*this, 
    "pairwise bmm", 
    transpose_reshape_cat_input->output, 
    unpacked_compressed_channels->output, 
    true, 
    false);
  layers.push_back(bmm);

  // flatten inner most 2 dimenions
  int l4_shape[2] = {batch_size, num_channels * compressed_num_channels};
  Reshape<3,2> *flattened_bmm = new Reshape<3,2>(*this, 
    "flattened_bmm", 
    bmm->output, 
    l4_shape);
  layers.push_back(flattened_bmm);

  // tanh
  Tanh<2> *tanh = new Tanh<2>(*this, 
    "tanh", 
    flattened_bmm->output, 
    l4_shape);
  layers.push_back(tanh);

  // final concat
  std::vector<Tensor> final_concats;
  final_concats.push_back(tanh->output);
  final_concats.push_back(dense_projection);


  Concat *cat_final = new Concat(*this, "concat_final", 2, &final_concats[0], 1);
  layers.push_back(cat_final);
  return cat_final->output;
}
