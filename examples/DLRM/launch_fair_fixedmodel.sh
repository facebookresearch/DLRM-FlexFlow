nnodes=$1
pngpu=8
mpirun -n $nnodes -N 1 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 ./run_fair_fixedmodel.sh $nnodes 
