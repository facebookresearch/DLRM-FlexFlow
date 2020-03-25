from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import itertools
import subprocess
import sys

NUM_EMB_OPS = 8
EMB_DIM = 64
HASH_SIZE = 1000000
NUM_INDICES = 1

repeat = 1

per_gpu_batch_sizes = [2 ** i for i in range(6, 14)]
# per_gpu_batch_sizes = [512]
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
                "./dlrm -ll:cpu 8 -ll:fsize 12000 -ll:zsize 20000 "
                "--arch-sparse-feature-size 64 "
                "--arch-embedding-size {emb_config} "
                "--arch-mlp-bot 64-512-512-64 "
                "--arch-mlp-top 576-1024-1024-1024-1 "
                "--epochs 20 --batch-size {batch_size} "
                "-dm:memorize -ll:gpu {gpus} -ll:util 12 -ll:dma 4 "
                "--strategy {strategy}"
            ).format(
                emb_config=emb_config,
                batch_size=batch_size,
                gpus=gpus,
                strategy=strategy,
            )

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
