#!/bin/bash

if [[ -z "${GASNET}"  ]]; then
    GASNET=$(find ${HOME}/. -name "GASNet-2019.9.0")
fi
echo "Using GASNET at - ${GASNET}"

#Bootstrap automake
./Bootstrap

cfgopts=" --enable-ibv \
        --enable-mpi \
        --disable-portals \
        --disable-mxm \
        --enable-pthreads \
        --enable-segment-fast \
        --enable-par \
        --disable-seq \
        --disable-parsync \
        --enable-mpi-compat \
        --with-ibv-spawner=mpi \
        --disable-ibv-rcv-thread \
        --disable-aligned-segments \
        --disable-pshm \
        --disable-fca \
        --enable-ibv-multirail \
        --with-ibv-max-hcas=4 \
        --with-ibv-physmem-max=2/3 \
        --disable-ibv-physmem-probe \
        --prefix=${PWD} "
#--enable-debug \
#--with-max-segsize=4GB/H \

echo "Using following configure option -- $cfgopts"

./configure $cfgopts

#build
#make par tests-par -j
make all -j
#make install

ln -fs $GASNET/ibv-conduit/libgasnet-ibv-par.a lib/.
