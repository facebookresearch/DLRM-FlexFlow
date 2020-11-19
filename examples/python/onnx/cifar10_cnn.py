from flexflow.core import *
from flexflow.keras.datasets import cifar10
from flexflow.onnx.model import ONNXModel

from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)
  
  dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
  input = ffmodel.create_tensor(dims_input, "", DataType.DT_FLOAT)

  # dims_label = [ffconfig.get_batch_size(), 1]
  # label = ffmodel.create_tensor(dims_label, "", DataType.DT_INT32)

  onnx_model = ONNXModel("cifar10_cnn.onnx")
  t = onnx_model.apply(ffmodel, {"input.1": input})

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label = ffmodel.get_label_tensor()
  
  use_external = True
  if (use_external == True):
    num_samples = 10000
    
    (x_train, y_train), (x_test, y_test) = cifar10.load_data(num_samples)
    
    x_train = x_train.astype('float32')
    x_train /= 255
    full_input_array = x_train
    print(full_input_array.__array_interface__["strides"])
    
    y_train = y_train.astype('int32')
    full_label_array = y_train
   
    print(full_input_array.__array_interface__["strides"])
    print(full_input_array.shape, full_label_array.shape)
    #print(full_input_array[0,:,:,:])
    #print(full_label_array[0, 0:64])
    print(full_label_array.__array_interface__["strides"])
    
    dims_full_input = [num_samples, 3, 32, 32]
    full_input = ffmodel.create_tensor(dims_full_input, "", DataType.DT_FLOAT)

    dims_full_label = [num_samples, 1]
    full_label = ffmodel.create_tensor(dims_full_label, "", DataType.DT_INT32)
    
    full_input.attach_numpy_array(ffconfig, full_input_array)
    full_label.attach_numpy_array(ffconfig, full_label_array)
    
    dataloader_input = SingleDataLoader(ffmodel, input, full_input, num_samples, DataType.DT_FLOAT)
    dataloader_label = SingleDataLoader(ffmodel, label, full_label, num_samples, DataType.DT_INT32)
    
    full_input.detach_numpy_array(ffconfig)
    full_label.detach_numpy_array(ffconfig)
    
    num_samples = dataloader_input.get_num_samples()
  else:
    # Data Loader
    dataloader = DataLoader4D(ffmodel, input, label, ffnetconfig=alexnetconfig)
    num_samples = dataloader.get_num_samples()

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()
  #epochs = 10

  ts_start = ffconfig.get_current_time()
  for epoch in range(0,epochs):
    if (use_external == True):
      dataloader_input.reset()
      dataloader_label.reset()
    else:
      dataloader.reset()
    ffmodel.reset_metrics()
    iterations = int(num_samples / ffconfig.get_batch_size())
    print(iterations, num_samples)

    for iter in range(0, int(iterations)):
      # if (len(alexnetconfig.dataset_path) == 0):
      #   if (iter == 0 and epoch == 0):
      #     dataloader.next_batch(ffmodel)
      # else:
    #  dataloader.next_batch(ffmodel)
      if (use_external == True):
        dataloader_input.next_batch(ffmodel)
        dataloader_label.next_batch(ffmodel)
      else:
        dataloader.next_batch(ffmodel)
      if (epoch > 0):
        ffconfig.begin_trace(111)
      ffmodel.forward()
      ffmodel.zero_gradients()
      ffmodel.backward()
      ffmodel.update()
      if (epoch > 0):
        ffconfig.end_trace(111)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));
  
  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < ModelAccuracy.CIFAR10_CNN.value:
    assert 0, 'Check Accuracy'

  conv_2d1 = ffmodel.get_layer_by_id(0)
  cbias_tensor = conv_2d1.get_input_tensor()
  #cbias_tensor = conv_2d1.get_output_tensor()
  cbias_tensor.inline_map(ffconfig)
  cbias = cbias_tensor.get_flat_array(ffconfig, DataType.DT_FLOAT)
  print(cbias.shape)
  print(cbias)
  cbias_tensor.inline_unmap(ffconfig)

  label.inline_map(ffconfig)
  label_array = label.get_flat_array(ffconfig, DataType.DT_INT32)
  print(label_array.shape)
  # print(cbias)
  print(label_array)
  label.inline_unmap(ffconfig)


if __name__ == "__main__":
  print("cifar10 cnn")
  top_level_task()
