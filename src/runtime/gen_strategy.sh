#!/bin/bash

if [ $# -ne 2 ]
then
	echo "Need two arguments"
	exit
fi

ngpus=$1
nembs=$2
python3 dlrm_strategy.py -f dlrm_strategy.cc -g ${ngpus} -e ${nembs}

echo "Compile..."
g++ dlrm_strategy.cc strategy.pb.cc -o generator -std=c++11 -L${PROTOBUF}/src/.libs -lprotobuf -L/usr/local/lib -I/usr/local/include -I${PROTOBUF}/src -pthread -O2

echo "Generate..."
./generator

echo "Done. dlrm_strategy_${nembs}embs_${ngpus}gpus.pb"
