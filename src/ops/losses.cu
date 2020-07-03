#include "model.h"
#include "cuda_helper.h"


LegionRuntime::Logger::Category loss_log("loss_functions");




template<int DIM>
void FFModel::sigmoid_cross_entropy_loss(
  const std::string& name,
  const Tensor& logits,
  const Tensor& labels
) 
{
  Loss<DIM>* op = new Loss<DIM>(
      *this, 
      name, 
      logits,
      labels,
      "sigmoid_cross_entropy_loss"
    );
  layers.push_back(op);
}


template<int DIM>
void FFModel::cross_entropy_loss(
  const std::string& name,
  const Tensor& logits,
  const Tensor& labels
) 
{
  Loss<DIM>* op = new Loss<DIM>(
      *this, 
      name, 
      logits,
      labels,
      "cross_entropy_loss"
    );
  layers.push_back(op);
}

template <int DIM>
Loss<DIM>::Loss(
  FFModel& model,
  const std::string& pc_name, 
  const Tensor& _logits,
  const Tensor& _labels,
  const std::string& _loss
) : Op(pc_name, _logits, _labels), profiling(model.config.profiling), loss(_loss) {
  task_is = IndexSpaceT<DIM>(model.get_or_create_task_is(DIM, pcname));
  model.create_data_parallel_partition_with_diff_dims<DIM, DIM>(
    _logits, task_is, input_lps[0], input_grad_lps[0]);

}

/*
loss (elementwise):
cross_entropy = sum(loss(x_i, y_i))
loss(x_i,y_i) = -y_i*log(sigmoid(x_i)) - (1-y_i)log(1-sigmoid(x_i))
          = max(x_i, 0) - x_i * y_i + log(1 + exp(-abs(x_i)))

derivative of cross entropy:
dL/dx = d(-y*log(x)-(1-y)log(1-x))/dx 
      = d(-y*log(x))/dx + d(-(1-y)log(1-x))/dx 
      = -y/x + (1-y)/(1-x) = (x-y)/x(1-x)

derivative of sigmoid cross entropy:
because x here is sigmoid(z), let z be logits
because derivative of sigmoid is:
  dx/dz = x(1-x)
apply chain rule we got
  dL/dz = dL/dx*dx/dz = ((x-y)/x(1-x)) * x(1-x) = x-y
*/

__global__
void sigmoid_crossentropy_with_logit(const float* logits,
                                    const float* labels,
                                    PerfMetrics* perf,
                                    int batch_size,
                                    float scale)
{
  CUDA_KERNEL_LOOP(i, batch_size)
  {
    float cross_entropy_loss = fmaxf(logits[i], 0) - logits[i] * labels[i] + log2f(1 + expf(-fabsf(logits[i])));
    atomicAdd(&(perf->train_loss), scale * cross_entropy_loss);
    atomicAdd(&(perf->train_all), 1);
  }
}





__global__
void sigmoid_crossentropy_with_logit_backward(float* logitsGrad,
                                              const float* logits,
                                              const float* labels,
                                              float factor,
                                              int size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    logitsGrad[i] = factor * (logits[i] - labels[i]);
  }
}



__global__
void binary_cross_entropy_forward_kernel(const float* pred,
                                    const float* labels,
                                    PerfMetrics* perf,
                                    int batch_size,
                                    float scale)
{
  CUDA_KERNEL_LOOP(i, batch_size)
  {
    // J = -(plogq + (1-p)log(1-q))
    float cross_entropy_loss = -(labels[i]*log2f(pred[i]) + (1-labels[i]) * log2f(1-pred[i]));
    atomicAdd(&(perf->train_loss), scale * cross_entropy_loss);
    atomicAdd(&(perf->train_all), 1);
  }
}

__global__
void binary_cross_entropy_backward_kernel(float* logitsGrad,
                                              const float* pred,
                                              const float* labels,
                                              float factor,
                                              int size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    logitsGrad[i] = (pred[i] - labels[i]) / (pred[i] * (1 - pred[i]));
  }
}







template <int DIM>
void Loss<DIM>::init(const FFModel& model)
{

}

template <int DIM>
void Loss<DIM>::forward(const FFModel& model)
{
}

template <int DIM>
void Loss<DIM>::backward(const FFModel& model) {
  ArgumentMap argmap;
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  IndexLauncher launcher(SIGMOID_CROSS_ENTROPY_WITH_LOGIT_BWD_TASK_ID, task_is,
                         TaskArgument(this, sizeof(Loss)), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         FFConfig::get_hash_id(std::string(name)));
  // regions[0]: _logit
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region));
  launcher.add_field(0, FID_DATA);
  // regions[1]: _label
  launcher.add_region_requirement(
      RegionRequirement(inputs[1].part, 0/*projection*/,
                      READ_ONLY, EXCLUSIVE, inputs[1].region));
  launcher.add_field(1, FID_DATA);
  // regions[2]: logit_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
  launcher.add_field(2, FID_DATA);
  FutureMap new_metrics = runtime->execute_index_space(ctx, launcher);
  // Update metrics
  TaskLauncher metrics_task(UPDATE_METRICS_TASK_ID, TaskArgument(NULL, 0));
  metrics_task.add_future(model.current_metrics);
  Rect<DIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  for (PointInRectIterator<DIM> it(part_rect); it(); it++) {
    metrics_task.add_future(new_metrics[*it]);
  }
  ((FFModel*)(&model))->current_metrics = runtime->execute_task(ctx, metrics_task);
}

template <int DIM>
PerfMetrics Loss<DIM>::backward_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime) {
    assert(regions.size() == 3);
    assert(task->regions.size() == 3);
    const Loss* op = (Loss*) task->args;
    TensorAccessorR<float, DIM> accLogits(
        regions[0], task->regions[0], FID_DATA, ctx, runtime);
    TensorAccessorR<float, DIM> accLabels(
        regions[1], task->regions[1], FID_DATA, ctx, runtime);
    TensorAccessorW<float, DIM> accLogitsGrad(
        regions[2], task->regions[2], FID_DATA, ctx, runtime, false/*readOutput*/);
    assert(accLogits.rect == accLabels.rect);
    assert(accLogits.rect == accLogitsGrad.rect);
    int batch_size = accLogits.rect.volume();
    // we sum the cross entropy of each data sample to get the cross entropy loss of the data set 
    float scale = 1.0f;

    if (op->profiling) {
      print_tensor<DIM, float>(accLabels.ptr, accLabels.rect, "[Loss:label]");
    }
    // Calculate loss
    PerfMetrics* perf;
    PerfMetrics perf_zc;
    perf_zc.train_loss = 0.0f;
    perf_zc.train_correct = perf_zc.train_all = 0;
    perf_zc.test_correct = perf_zc.test_all = 0;
    perf_zc.val_correct = perf_zc.val_all = 0;
  
    checkCUDA(cudaMalloc(&perf, sizeof(PerfMetrics)));
    checkCUDA(cudaMemcpy(perf, &perf_zc, sizeof(PerfMetrics), cudaMemcpyHostToDevice));
    // calculate loss store in perf
    if (op->loss == "sigmoid_cross_entropy_loss") {
      sigmoid_crossentropy_with_logit<<<GET_BLOCKS(accLogits.rect.volume()), CUDA_NUM_THREADS>>>(accLogits.ptr,
        accLogits.ptr,
        perf,
        batch_size,
        scale);
    } else if (op->loss == "cross_entropy_loss") {
      binary_cross_entropy_forward_kernel<<<GET_BLOCKS(accLogits.rect.volume()), CUDA_NUM_THREADS>>>(accLogits.ptr,
        accLogits.ptr,
        perf,
        batch_size,
        scale);
    } else {
      std::cout << "unknown loss " << op->loss << std::endl;
       throw 991;
    }



    checkCUDA(cudaMemcpy(&perf_zc, perf, sizeof(PerfMetrics), cudaMemcpyDeviceToHost));
    checkCUDA(cudaFree(perf));
    // Calculate backward

    if (op->loss == "sigmoid_cross_entropy_loss") {
      sigmoid_crossentropy_with_logit_backward<<<GET_BLOCKS(accLogits.rect.volume()), CUDA_NUM_THREADS>>>(
        accLogitsGrad.ptr, 
        accLogits.ptr, 
        accLabels.ptr,
        scale, 
        accLogits.rect.volume()
      );
    } else if (op->loss == "cross_entropy_loss") {
      binary_cross_entropy_backward_kernel<<<GET_BLOCKS(accLogits.rect.volume()), CUDA_NUM_THREADS>>>(
        accLogitsGrad.ptr, 
        accLogits.ptr, 
        accLabels.ptr,
        scale, 
        accLogits.rect.volume()
      );
    } else {
      std::cout << "unknown loss " << op->loss << std::endl;
      throw 991;
    }





    checkCUDA(cudaDeviceSynchronize());
    if (op->profiling) {
      print_tensor<DIM, float>(accLogits.ptr, accLogits.rect, "[Loss:logit]");
      print_tensor<DIM, float>(accLogitsGrad.ptr, accLogitsGrad.rect, "[Loss:logit_grad]");
    }
    return perf_zc; 
}










template void FFModel::sigmoid_cross_entropy_loss<2>(
  const std::string& name,
  const Tensor& logits,
  const Tensor& labels);

template void FFModel::cross_entropy_loss<2>(
  const std::string& name,
  const Tensor& logits,
  const Tensor& labels);

template Loss<2>::Loss(
  FFModel& model,
  const std::string& pc_name, 
  const Tensor& _logits,
  const Tensor& _labels,
  const std::string& _loss
);

template void Loss<2>::init(const FFModel& model);

template void Loss<2>::forward(const FFModel& model);

template void Loss<2>::backward(const FFModel& model);

template PerfMetrics Loss<2>::backward_task(
  const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx, Runtime *runtime);
