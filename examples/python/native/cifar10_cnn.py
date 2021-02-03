from flexflow.core import *
from flexflow.keras.datasets import cifar10

from accuracy import ModelAccuracy

def top_level_task():
  ffconfig = FFConfig()
  alexnetconfig = NetConfig()
  print(alexnetconfig.dataset_path)
  ffconfig.parse_args()
  print("Python API batchSize(%d) workersPerNodes(%d) numNodes(%d)" %(ffconfig.get_batch_size(), ffconfig.get_workers_per_node(), ffconfig.get_num_nodes()))
  ffmodel = FFModel(ffconfig)

  dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
  input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

  t = ffmodel.conv2d(input_tensor, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.conv2d(t, 32, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.pool2d(t, 2, 2, 2, 2, 0, 0,)
  t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.conv2d(t, 64, 3, 3, 1, 1, 1, 1, ActiMode.AC_MODE_RELU)
  t = ffmodel.pool2d(t, 2, 2, 2, 2, 0, 0)
  t = ffmodel.flat(t);
  t = ffmodel.dense(t, 512, ActiMode.AC_MODE_RELU)
  t = ffmodel.dense(t, 10)
  t = ffmodel.softmax(t)

  ffoptimizer = SGDOptimizer(ffmodel, 0.01)
  ffmodel.set_sgd_optimizer(ffoptimizer)
  ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
  label_tensor = ffmodel.get_label_tensor()

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

  dataloader_input = ffmodel.create_data_loader(input_tensor, full_input_array)
  dataloader_label = ffmodel.create_data_loader(label_tensor, full_label_array)

  num_samples = dataloader_input.get_num_samples()

  ffmodel.init_layers()

  epochs = ffconfig.get_epochs()

  ts_start = ffconfig.get_current_time()

  ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

  ts_end = ffconfig.get_current_time()
  run_time = 1e-6 * (ts_end - ts_start);
  print("epochs %d, ELAPSED TIME = %.4fs, THROUGHPUT = %.2f samples/s\n" %(epochs, run_time, num_samples * epochs / run_time));

  perf_metrics = ffmodel.get_perf_metrics()
  accuracy = perf_metrics.get_accuracy()
  if accuracy < ModelAccuracy.CIFAR10_CNN.value:
    assert 0, 'Check Accuracy'

if __name__ == "__main__":
  print("cifar10 cnn")
  top_level_task()
