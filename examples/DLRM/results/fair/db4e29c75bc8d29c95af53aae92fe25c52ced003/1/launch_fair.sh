runscript=$1
nnodes=$2
pnemb=8
mpirun -n $nnodes -N 1 --mca btl_openib_allow_ib 1 --mca btl_openib_warn_default_gid_prefix 0 $runscript $nnodes $pnemb
