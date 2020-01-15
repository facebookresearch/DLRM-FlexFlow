#INSTRUCTIONS# Conduit-specific Makefile fragment settings
#INSTRUCTIONS#
#INSTRUCTIONS# The contents of this file are embedded into the 
#INSTRUCTIONS# *-(seq,par,parsync).mak Makefile fragments at conduit build time
#INSTRUCTIONS# The settings in those fragments are used to build GASNet clients
#INSTRUCTIONS# (including the GASNet tests). 
#INSTRUCTIONS# See the conduit-writer instructions in the generated fragments
#INSTRUCTIONS# or $(top_srcdir)/other/fragment-head.mak.in for usage info.

# AMMPI is MPI-based, which requires us to link using the system MPI compiler
GASNET_LD_OVERRIDE = /usr/bin/mpicc
GASNET_LDFLAGS_OVERRIDE = -D_GNU_SOURCE=1 -O3 --param max-inline-insns-single=35000 --param inline-unit-growth=10000 --param large-function-growth=200000  -Wno-unused -Wunused-result -Wno-unused-parameter -Wno-address  

# Linker feature requirements embedded in GASNET_LD(FLAGS) which are not satisfied solely by GASNET_LIBS 
# (eg possible dependence on implicit MPI or C++ libraries added by a linker wrapper in GASNET_LD):
GASNET_LD_REQUIRES_MPI = 1

# hooks for using AMMPI library from within build tree ###NOINSTALL### 
# (nothing additional required for installed copy)     ###NOINSTALL###
CONDUIT_INCLUDES = -I/private/home/ehsanardestani/DLRM_FlexFlow_d/GASNet-2019.9.0/other/ammpi -I/private/home/ehsanardestani/DLRM_FlexFlow_d/GASNet-2019.9.0/other/ammpi ###NOINSTALL###
CONDUIT_LIBDIRS =  -L/private/home/ehsanardestani/DLRM_FlexFlow_d/GASNet-2019.9.0/other/ammpi        ###NOINSTALL###

CONDUIT_LIBS = $(CONDUIT_LIBDIRS) -lammpi 
