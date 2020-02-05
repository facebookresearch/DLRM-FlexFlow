
void DataLoader::load_batched_matrices(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime)
{
    int num_dim = 3;
  assert(regions.size() == num_dim);
  assert(task->regions.size() == num_dim);
  SampleIdxs* meta = (SampleIdxs*) task->local_args;
  TensorAccessorR<float, num_dim> acc_full_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, num_dim> acc_batch_input(
      regions[1], task->regions[1], FID_DATA, ctx, runtime,
      false/*readOutput*/);
  int batch_size = acc_batch_input.rect.hi[1] - acc_batch_input.rect.lo[1] + 1;
  int num_feats = acc_batch_input.rect.hi[0] - acc_batch_input.rect.lo[0] + 1;
  assert(acc_batch_input.rect.hi[0] == acc_full_input.rect.hi[0]);
  assert(acc_batch_input.rect.lo[0] == acc_full_input.rect.lo[0]);
  float* input_zc;
  checkCUDA(cudaHostAlloc(&input_zc, sizeof(float) * acc_batch_input.rect.volume(),
                          cudaHostAllocPortable | cudaHostAllocMapped));
  assert(batch_size == meta->num_samples);
  for (int i = 0; i < batch_size; i++) {
    int base_offset = meta->idxs[i] * num_feats;
    for (int j = 0; j < num_feats; j++)
      input_zc[i*num_feats+j] = acc_full_input.ptr[base_offset+j];
  }
  checkCUDA(cudaMemcpy(acc_batch_input.ptr, input_zc,
                       sizeof(float) * acc_batch_input.rect.volume(),
                       cudaMemcpyHostToDevice));
  checkCUDA(cudaFreeHost(input_zc));
}
