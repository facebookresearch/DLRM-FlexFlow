#INSTRUCTIONS# Conduit-specific Makefile fragment settings
#INSTRUCTIONS#
#INSTRUCTIONS# The contents of this file are embedded into the 
#INSTRUCTIONS# *-(seq,par,parsync).mak Makefile fragments at conduit build time
#INSTRUCTIONS# The settings in those fragments are used to build GASNet clients
#INSTRUCTIONS# (including the GASNet tests). 
#INSTRUCTIONS# See the conduit-writer instructions in the generated fragments
#INSTRUCTIONS# or $(top_srcdir)/other/fragment-head.mak.in for usage info.

# When ibv-conduit uses an MPI-based bootstrapper, we must
# link using the system MPI compiler
GASNET_LD_OVERRIDE = /usr/bin/mpicc
GASNET_LDFLAGS_OVERRIDE = -D_GNU_SOURCE=1 -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000  -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address  
MPI_COMPAT_LIBS = 

# Linker feature requirements embedded in GASNET_LD(FLAGS) which are not satisfied solely by GASNET_LIBS 
# (eg possible dependence on implicit MPI or C++ libraries added by a linker wrapper in GASNET_LD):
GASNET_LD_REQUIRES_MPI = 1

# Some platforms need extra -libs for the socket calls in ssh-spawner:
SSH_LIBS = 

CONDUIT_INCLUDES = -I/private/home/ehsanardestani/DLRM_FlexFlow_d/GASNet-2019.9.0/other/firehose ###NOINSTALL###

CONDUIT_LDFLAGS =   
CONDUIT_LIBS = -libverbs $(MPI_COMPAT_LIBS) $(SSH_LIBS)  

# If ibv-conduit has internal conduit threads, then it needs 
# threading flags and libs - even in GASNET_SEQ mode
CONDUIT_DEFINES_SEQ = -D_REENTRANT
CONDUIT_LIBS_SEQ =    -lpthread

# Clients may want/need to know which spawners we support:
GASNET_SPAWNER_DEFAULT = mpi
#GASNET_SPAWNER_PMI = 1
GASNET_SPAWNER_PMI = 0
GASNET_SPAWNER_MPI = 1
#GASNET_SPAWNER_MPI = 0
GASNET_SPAWNER_SSH = 1
#GASNET_SPAWNER_SSH = 0
