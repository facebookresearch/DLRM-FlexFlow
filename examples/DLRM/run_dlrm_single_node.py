from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import itertools
import subprocess
import sys

NUM_EMB_OPS = 8
EMB_DIM = 64
HASH_SIZE = 3000000
POOLING_FACTOR = 38

# TOP_MLP_SIZE = 1024
# NUM_TOP_MLPS = 9
# TOP_MLP_SIZE = 4096
# NUM_TOP_MLPS = 8

TOP_CONFIGS  = [(1024,9), (4096,8)]

DENSE_FEATURES = 1024
BOT_MLP_SIZE = 1024
NUM_BOT_MLPS = 3
NUM_EMBEDDING = 1
repeat = 1

per_gpu_batch_sizes = [2 ** i for i in range(6, 14)]
# per_gpu_batch_sizes = [512]
num_gpus = [1, 2, 4, 8]

with open("res_single_node.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    write_cmd = 1
    for top_c in TOP_CONFIGS:
        TOP_MLP_SIZE, NUM_TOP_MLPS = top_c
        for nodes in [1]:  
          for gpu_batch_size, gpus in itertools.product(per_gpu_batch_sizes, num_gpus):
              for _ in range(repeat):
                  strategy = "../../src/runtime/dlrm_strategy_emb_{}_gpu_{}_node_{}.pb".format(NUM_EMBEDDING, gpus, nodes)

                  batch_size = gpu_batch_size * gpus
                  print("==== Running with {} gpus, batch size of {}, per-gpu batch size of {} TOP_MLP_SIZE {} num_top_mlp {} ====".format(
                      gpus, batch_size, gpu_batch_size, TOP_MLP_SIZE, NUM_TOP_MLPS
                  ))

                  emb_config = "-{}".format(HASH_SIZE) * NUM_EMB_OPS
                  emb_config = emb_config[1:]

                  bot_mlp_config = (
                      "{}".format(DENSE_FEATURES)
                      + "-{}".format(BOT_MLP_SIZE) * (NUM_BOT_MLPS - 1)
                      + "-{}".format(EMB_DIM)
                  )

                  top_mlp_config = (
                      "{}".format((NUM_EMB_OPS + 1) * EMB_DIM)
                      + "-{}".format(TOP_MLP_SIZE) * (NUM_TOP_MLPS - 1)
                      + "-1"
                  )

                  cmd = (
                      "./dlrm -ll:cpu 8 -ll:fsize 14000 -ll:zsize 20000 "
                      "--arch-sparse-feature-size 64 "
                      "--arch-embedding-size {emb_config} "
                      "--embedding-bag-size {pooling_factor} "
                      "--arch-mlp-bot {bot_mlp_config} "
                      "--arch-mlp-top {top_mlp_config} "
                      "--epochs 20 --batch-size {batch_size} "
                      "--data-size {data_size} "
                      "-dm:memorize -ll:gpu {gpus} -ll:util 12 -ll:dma 4 "
                      "--strategy {strategy} "
                      # "-lg:prof 1 -lg:prof_logfile prof_{gpus}_{gpu_batch_size}.gz"
                  ).format(
                      emb_config=emb_config,
                      pooling_factor=POOLING_FACTOR,
                      bot_mlp_config=bot_mlp_config,
                      top_mlp_config=top_mlp_config,
                      batch_size=batch_size,
                      data_size=batch_size * 4,  # make sure we have data for 4 iterations
                      gpus=gpus,
                      strategy=strategy,
                      gpu_batch_size=gpu_batch_size,
                  )

                  print(cmd)
                  if write_cmd:
                      writer.writerow(cmd)
                      write_cmd = 0

                  try:
                      gen_strategy_cmd = 'cd ../../src/runtime && ./gen_strategy.sh %d %d %d' % (gpus, NUM_EMBEDDING, nodes)
                      st_out = subprocess.check_output(gen_strategy_cmd, stderr=subprocess.STDOUT, shell=True)
                      print(st_out)
                      out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
                      res = out.splitlines()[-1]
                      throughput = res.strip().split()[-2]
                      print(res.decode("utf-8"))
                  except KeyboardInterrupt:
                      sys.exit(0)
                  except Exception as e:
                      throughput = b"ERROR"
                      print("An error occured.\n%s" % (e))

                  writer.writerow([str(gpus), str(gpu_batch_size), throughput.decode("utf-8"), str(NUM_TOP_MLPS), str(TOP_MLP_SIZE)])