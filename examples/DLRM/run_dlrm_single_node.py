from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import itertools
import subprocess
import sys

NUM_EMB_OPS = 8
EMB_DIM = 64
HASH_SIZE = 3000000
POOLING_FACTOR = 100

repeat = 1

per_gpu_batch_sizes = [2 ** i for i in range(6, 14)]
num_gpus = [1, 2, 4, 8]

with open("res_single_node.csv", "wb") as csvfile:
    writer = csv.writer(csvfile)

    for gpu_batch_size, gpus in itertools.product(per_gpu_batch_sizes, num_gpus):
        for _ in range(repeat):
            strategy = "../../src/runtime/dlrm_strategy_emb_8_gpu_{}_node_1.pb".format(gpus)

            batch_size = gpu_batch_size * gpus
            print("==== Running with {} gpus, batch size of {}, per-gpu batch size of {} ====".format(
                gpus, batch_size, gpu_batch_size
            ))

            emb_config = "-{}".format(HASH_SIZE) * NUM_EMB_OPS
            emb_config = emb_config[1:]

            cmd = (
                "./dlrm -ll:cpu 8 -ll:fsize 14000 -ll:zsize 20000 "
                "--arch-sparse-feature-size 64 "
                "--arch-embedding-size {emb_config} "
                "--embedding-bag-size {pooling_factor} "
                "--arch-mlp-bot 1024-1024-1024-64 "
                "--arch-mlp-top 576-1024-1024-1024-1024-1024-1 "
                "--epochs 20 --batch-size {batch_size} "
                "--data-size {data_size} "
                "-dm:memorize -ll:gpu {gpus} -ll:util 12 -ll:dma 4 "
                "--strategy {strategy}"
            ).format(
                emb_config=emb_config,
                pooling_factor=POOLING_FACTOR,
                batch_size=batch_size,
                data_size=batch_size * 4,  # make sure we have data for 4 iterations
                gpus=gpus,
                strategy=strategy,
            )

            print(cmd)

            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
                res = out.splitlines()[-1]
                throughput = res.strip().split()[-2]
                print(res)
            except KeyboardInterrupt:
                sys.exit(0)
            except:
                throughput = "ERROR"
                print("An error occured.")

            writer.writerow([str(gpus), str(gpu_batch_size), throughput])
