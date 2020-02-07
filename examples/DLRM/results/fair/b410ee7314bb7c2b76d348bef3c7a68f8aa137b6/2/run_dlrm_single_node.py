from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import itertools
import subprocess
import sys


repeat = 1

per_gpu_batch_sizes = [2 ** i for i in range(6, 14)]
# per_gpu_batch_sizes = [512]
num_gpus = [1, 2, 4, 8]

with open("res_single_node.csv", "wb") as csvfile:
    writer = csv.writer(csvfile)

    for gpus, gpu_batch_size in itertools.product(num_gpus, per_gpu_batch_sizes):
        for _ in range(repeat):
            strategy = "../../src/runtime/dlrm_strategy_emb_8_gpu_{}_node_1.pb".format(gpus)
            batch_size = gpu_batch_size * gpus

            print("==== Running with {} gpus, batch size of {}, per-gpu batch size of {} ====".format(
                gpus, batch_size, batch_size // gpus
            ))

            cmd = (
                "./dlrm -ll:cpu 8 -ll:fsize 12000 -ll:zsize 20000 "
                "--arch-sparse-feature-size 64 "
                "--arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 "
                "--arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 20 "
                "-dm:memorize -ll:gpu {gpus} -ll:util 12 -ll:dma 4 --batch-size {} --strategy {}"
            ).format(batch_size, strategy, gpus=gpus)

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
